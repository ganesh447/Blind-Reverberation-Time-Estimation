"""Generate noisy reverberant training utterances as FLOAT WAVs.

Run:  awssdr-utterances --root /path/to/corpora --rir-dir /path/to/rirs
Then: awssdr-decay-rates --utterances /path/to/corpora/training_utterances --out training_sdrs

--root holds Speech/, Noise/ and Eval_data/. --rir-dir holds the RIR .wav files and T30_ground_truth.csv.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import multiprocessing
import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from functools import partial
from pathlib import Path
from typing import NamedTuple, Self

import librosa
import numpy as np
import soundfile as sf
from pyvoicebox import v_activlev
from scipy.signal import fftconvolve, istft, stft

from awssdr.data.decay_rates import MIN_SAMPLES
from awssdr.dataset_layout import FS, RIR_LABELS_CSV

# Replace these two placeholders with the folders that hold your inputs, or pass --root / --rir-dir.
DEFAULT_ROOT = Path("PATH/TO/YOUR/CORPORA_FOLDER")  # contains Speech/, Noise/ and Eval_data/
DEFAULT_RIR_DIR = Path("PATH/TO/YOUR/RIR_FOLDER")  # RIR .wav files and T30_ground_truth.csv

RECIPE_VERSION = "v5"  # bump whenever the recipe changes how the audio is made

NOISE_EXCERPT_SAMPLES = 10 * FS
NPERSEG = 1024
STREAM_COUNT = (4, 7)
TRANSIENT_RATE_HZ = 0.05  # mean AID events per second of bed (~1 per 20 s), so density does not depend on utterance length; a round occasional-event rate, not fitted to ACE
TRANSIENT_GAIN_DB = (-5.0, 10.0)
BABBLE_BED_DB = -15.0
BABBLE_BED_RECORDINGS = "Ambient"  # babble's ambient bed is shaped from these recordings

PASSES = ("ears", "ears", "ears", "gsp", "qualcomm", "st_aed", "timit")
SOURCES = tuple(dict.fromkeys(PASSES))
NOISE_TYPES = ("Ambient", "Fan", "Babble")
CONDITIONS = tuple(itertools.product(NOISE_TYPES, (0, 10, 20)))
DEMAND_ENVS = ("SPSQUARE", "TMETRO", "OHALLWAY", "PSTATION", "STRAFFIC", "NRIVER",
               "PCAFETER", "OMEETING", "DLIVING", "NPARK", "PRESTO", "DKITCHEN")
TRANSIENT_CLASSES = ("foot_steps", "cough", "paper", "book", "keys", "pen_case", "scissors",
                     "rubber_band", "roll_stool", "zipper", "snapping", "clapping")
# Libraries whose versions define the audio, recorded in run.json (pyvoicebox-sap defines the P.56 level).
DEPENDENCIES = ("pyvoicebox-sap", "numpy", "scipy", "librosa", "soundfile")


class Rir(NamedTuple):
    """A room impulse response file and its T30 label in seconds."""
    path: Path
    t30: float


@dataclass(frozen=True)
class Sample:
    """One planned utterance: target speech, RIR, T30 label and noise condition."""
    sample_id: str
    pass_index: int
    speech_source: str
    rir_path: Path
    speech_path: Path
    target_t30: float
    noise_type: str
    snr_db: int
    noise_seed: tuple[int, int, int, int]  # (seed, pass, RIR, condition) indices; seeds the noise RNG


@dataclass(frozen=True)
class Transient:
    """A trimmed AID event clip and its active RMS."""
    audio: np.ndarray
    rms: float
    path: Path


@dataclass(frozen=True)
class MixLevels:
    """Levels behind one mixture: P.56 speech power, noise power, applied noise gain, pre-normalisation peak."""
    speech_active_power: float
    noise_power: float
    noise_gain: float
    mixture_peak: float


def load_audio(path: Path) -> np.ndarray:
    """Load non-silent mono float32 audio at FS."""
    audio, _ = librosa.load(path, sr=FS, mono=True)
    audio = audio.astype(np.float32, copy=False)
    if not np.isfinite(audio).all() or not np.any(audio):
        raise ValueError(f"{path}: expected finite, non-silent audio")
    return audio


def load_rirs(directory: Path) -> list[Rir]:
    """Pair sorted RIR WAVs with T30_ground_truth.csv labels."""
    with (directory / RIR_LABELS_CSV).open(newline="") as handle:
        labels = {row["filename"]: float(row["t30"]) for row in csv.DictReader(handle)}
    paths = sorted(directory.glob("*.wav"))
    if not paths or set(labels) != {path.name for path in paths}:
        raise ValueError(f"RIR WAVs and T30_ground_truth.csv rows must match: {directory}")
    if not all(np.isfinite(t30) and t30 > 0 for t30 in labels.values()):
        raise ValueError("all T30 labels must be finite and positive")
    return [Rir(path, labels[path.name]) for path in paths]


def _ears(speech: Path) -> list[Path]:
    return [p for p in sorted((speech / "EARS").glob("**/*.wav"))
            if p.name.startswith(("rainbow_", "sentences_", "freeform_speech_"))
            or (p.name.startswith("emo_") and p.name.endswith("_sentences.wav"))]


def _gsp(speech: Path) -> list[Path]:
    directory = speech / "SET-2" / "GSP"
    # clips in the validation and testing lists are held out, so they never appear in training
    held_out = {line.strip() for name in ("validation_list.txt", "testing_list.txt")
                for line in (directory / name).read_text().splitlines() if line.strip()}
    return [p for p in sorted(directory.glob("*/*.wav"))
            if p.parent.name != "_background_noise_"
            and p.relative_to(directory).as_posix() not in held_out]


def _qualcomm(speech: Path) -> list[Path]:
    return sorted((speech / "SET-2" / "Qualcomm").glob("*/*/*.wav"))


def _st_aed(speech: Path) -> list[Path]:
    return [p for p in sorted((speech / "SET-2" / "ST_AED").glob("*.wav"))
            if re.match(r"[fm]\d+_", p.name)]


def _timit(speech: Path) -> list[Path]:
    # SI and SX sentences only; the SA sentences are read by every speaker
    return [p for p in sorted((speech / "timit" / "timit").glob("dr*-*/*.wav"))
            if p.stem[:2].lower() in ("si", "sx")]


SPEECH_FINDERS = {"ears": _ears, "gsp": _gsp, "qualcomm": _qualcomm, "st_aed": _st_aed, "timit": _timit}


def find_speech(root: Path, source: str) -> tuple[Path, ...]:
    """List one corpus's training clips under ROOT/Speech."""
    paths = SPEECH_FINDERS[source](root / "Speech")
    if not paths:
        raise FileNotFoundError(f"no {source} speech clips under {root / 'Speech'}")
    return tuple(paths)


def load_transient(path: Path) -> Transient:
    """Trim an AID clip at 30 dB and measure RMS over samples above 5 % of its peak."""
    audio, _ = librosa.effects.trim(load_audio(path), top_db=30)
    active = np.abs(audio) > 0.05 * np.max(np.abs(audio))
    rms = float(np.sqrt(np.mean(np.asarray(audio[active], dtype=np.float64) ** 2)))
    return Transient(audio, rms, path)


def checked_recording_paths(root: Path, noise_types: tuple[str, ...]) -> dict[str, tuple[Path, ...]]:
    """Check required recordings; Babble uses BABBLE_BED_RECORDINGS for its bed."""
    needed = {BABBLE_BED_RECORDINGS if kind == "Babble" else kind for kind in noise_types}
    recordings = {}
    if "Ambient" in needed:
        recordings["Ambient"] = tuple(root / "Noise" / "DEMAND" / f"{env}_16k" / env / "ch01.wav"
                                      for env in DEMAND_ENVS)
    if "Fan" in needed:
        recordings["Fan"] = tuple(root / "Noise" / "MS-SNSD" / f"AirConditioner_{i}.wav"
                                  for i in range(1, 11))
    for path in itertools.chain.from_iterable(recordings.values()):
        if not path.is_file():
            raise FileNotFoundError(f"missing noise recording: {path}")
        if sf.info(path).duration * FS < NPERSEG:
            raise ValueError(f"noise recording shorter than {NPERSEG} samples: {path}")
    return recordings


def load_aid_transients(root: Path) -> tuple[Transient, ...]:
    """Load every AID clip of each transient class."""
    aid_paths = sorted((root / "Eval_data" / "AID" / "wavs").glob("*.wav"))
    transients = []
    for cls in TRANSIENT_CLASSES:
        matches = [p for p in aid_paths if re.fullmatch(rf"{cls}_\d+_SH_MKH800\.wav", p.name)]
        if not matches:
            raise FileNotFoundError(f"no AID transients for class {cls}")
        transients.extend(load_transient(path) for path in matches)
    return tuple(transients)


def _clip_cycle(paths: tuple[Path, ...], rng: np.random.Generator) -> Iterator[Path]:
    """Yield clips forever, reshuffling after each full pass."""
    while True:
        for i in rng.permutation(len(paths)):
            yield paths[i]


def _rir_tag(rir: Path) -> str:
    """Filename-safe RIR stem plus a short hash, so truncated stems stay unique."""
    stem = re.sub(r"[^A-Za-z0-9_.-]", "-", rir.stem)[:64]
    return f"{stem}_{hashlib.blake2b(rir.name.encode(), digest_size=4).hexdigest()}"


def plan_samples(speech: dict[str, tuple[Path, ...]], rirs: list[Rir], seed: int,
                 sources: tuple[str, ...], noise_types: tuple[str, ...],
                 ears_passes: int = PASSES.count("ears")) -> list[Sample]:
    """
    Assign speech and noise seeds to pass x RIR x condition slots.

    Draw before filtering so restricted runs retain the full run's clips and
    (seed, pass, RIR, condition) noise seeds. Adding or removing RIR files
    changes later indices; --limit-rirs only keeps the first RIRs.
    Skipped EARS repeats keep the other passes' indices, so fewer EARS passes give a subset of the full run.
    """
    samples = []
    for index, source in enumerate(PASSES):
        if source not in sources or (source == "ears" and PASSES[:index].count("ears") >= ears_passes):
            continue
        clips = _clip_cycle(speech[source], np.random.default_rng([seed, index]))
        for rir_index, rir in enumerate(rirs):
            for condition_index, (noise_type, snr_db) in enumerate(CONDITIONS):
                clip = next(clips)  # drawn before filtering, so subsets match the full run
                if noise_type not in noise_types:
                    continue
                sample_id = f"p{index:02d}_{source}__{_rir_tag(rir.path)}__{noise_type}__{snr_db}dB"
                samples.append(Sample(sample_id, index, source, rir.path, clip, rir.t30,
                                      noise_type, snr_db, (seed, index, rir_index, condition_index)))
    if not samples:
        raise ValueError("the selected sources, noise types and RIRs yield no samples")
    return samples


def read_excerpt(rng: np.random.Generator, paths: tuple[Path, ...]) -> tuple[np.ndarray, dict]:
    """Read a random excerpt of at most NOISE_EXCERPT_SAMPLES from one of `paths`, resampled to FS."""
    path = paths[rng.integers(len(paths))]
    info = sf.info(path)
    length = int(info.frames * FS / info.samplerate)  # in FS-rate samples, whatever the file's own rate
    count = min(NOISE_EXCERPT_SAMPLES, length)
    start = int(rng.integers(length - count + 1))
    # offset/duration read only the excerpt, so long recordings are not loaded in full
    excerpt, _ = librosa.load(path, sr=FS, mono=True, offset=start / FS, duration=count / FS)
    return excerpt, {"path": str(path), "offset_samples_16k": start, "samples": len(excerpt)}


def shaped_noise(rng: np.random.Generator, excerpt: np.ndarray, n: int) -> np.ndarray:
    """Shape `n` samples of white Gaussian noise with the excerpt's mean STFT magnitude."""
    if min(n, len(excerpt)) < NPERSEG:
        raise ValueError(f"noise shaping needs at least {NPERSEG} samples")
    _, _, spectrum = stft(excerpt, FS, nperseg=NPERSEG)
    # RMS magnitude per frequency bin over time: the excerpt's spectral colour; its timing is discarded
    envelope = np.sqrt(np.mean(np.abs(spectrum) ** 2, axis=1))
    _, _, white = stft(rng.standard_normal(n), FS, nperseg=NPERSEG)
    _, bed = istft(white * envelope[:, None], FS, nperseg=NPERSEG)
    return np.pad(bed, (0, max(0, n - len(bed))))[:n]


def add_transients(rng: np.random.Generator, bed: np.ndarray,
                   transients: tuple[Transient, ...]) -> list[dict]:
    """Add random transients to `bed` in place and describe each event."""
    n = len(bed)
    bed_rms = float(np.sqrt(np.mean(bed ** 2)))
    events = []
    for _ in range(rng.poisson(TRANSIENT_RATE_HZ * n / FS)):
        event = transients[rng.integers(len(transients))]
        relative_db = float(rng.uniform(*TRANSIENT_GAIN_DB))
        gain = bed_rms / event.rms * 10 ** (relative_db / 20)  # event's active RMS ends up relative_db re the bed
        offset = int(rng.integers(max(n - len(event.audio), 0) + 1))  # an event longer than the bed is cut
        segment = event.audio[:n - offset]
        bed[offset:offset + len(segment)] += gain * segment
        events.append({"path": str(event.path), "offset_samples": offset, "gain": gain,
                       "relative_db": relative_db, "used_samples": len(segment)})
    return events


def load_babble_speakers(root: Path) -> dict[Path, tuple[Path, ...]]:
    """
    Group conversational EARS clips by speaker, excluding stylized readings.
    Require enough speakers to exclude the target's own speaker.
    """
    clips = [path for path in find_speech(root, "ears")
             if path.name.startswith("freeform_speech_") or path.stem.endswith("_regular")]
    speakers = {folder: tuple(group)
                for folder, group in itertools.groupby(sorted(clips), key=lambda path: path.parent)}
    if len(speakers) <= STREAM_COUNT[1]:
        raise ValueError(f"Babble needs {STREAM_COUNT[1] + 1} EARS speakers with conversational clips, "
                         f"found {len(speakers)}")
    return speakers


def babble(rng: np.random.Generator, speakers: dict[Path, tuple[Path, ...]], target: Path,
           n: int, rir: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    """
    Sum a STREAM_COUNT number of distinct speakers at equal levels, excluding the target speaker.

    `speakers` must come from `load_babble_speakers`, which guarantees enough speakers.

    Each cropped stream is set to unit P.56 active level after the room response.
    Clips shorter than `n` are tiled; longer clips are cropped at a random offset.
    """
    others = [folder for folder in speakers if folder != target.parent]
    mixture = np.zeros(n)
    streams = []
    count = int(rng.integers(STREAM_COUNT[0], STREAM_COUNT[1] + 1))
    for index in rng.choice(len(others), size=count, replace=False):
        clips = speakers[others[index]]
        path = clips[rng.integers(len(clips))]
        clip = load_audio(path)
        tiled = len(clip) < n
        offset = int(rng.integers(len(clip) if tiled else len(clip) - n + 1))
        source = np.tile(clip, n // len(clip) + 2) if tiled else clip  # +2: the offset still leaves n samples
        segment = source[offset:offset + n]
        try:
            gain = 1 / np.sqrt(active_speech_power(fftconvolve(segment, rir)[:n]))
        except ValueError as exc:
            raise ValueError(f"invalid Babble excerpt: {path}") from exc
        mixture += segment * gain
        streams.append({"path": str(path), "speaker": path.parent.name,
                        "offset_samples": offset, "tiled": tiled, "gain": gain})
    return mixture, streams


@dataclass(frozen=True)
class NoiseBank:
    """Noise recordings per bed type, babble clips per EARS speaker, and transient events."""
    recordings: dict[str, tuple[Path, ...]]
    babble_speakers: dict[Path, tuple[Path, ...]]
    transients: tuple[Transient, ...]

    @classmethod
    def load(cls, root: Path, noise_types: tuple[str, ...]) -> Self:
        """Validate and load everything the noise generator reads for `noise_types`."""
        speakers = load_babble_speakers(root) if "Babble" in noise_types else {}
        return cls(checked_recording_paths(root, noise_types), speakers, load_aid_transients(root))

    def recording_bed(self, rng: np.random.Generator, kind: str, n: int) -> tuple[np.ndarray, dict]:
        """Shape noise like a random `kind` recording excerpt and add transients to it."""
        excerpt, recording = read_excerpt(rng, self.recordings[kind])
        bed = shaped_noise(rng, excerpt, n)
        return bed, {"recording": recording, "transients": add_transients(rng, bed, self.transients)}

    def babble_bed(self, rng: np.random.Generator, target: Path, n: int,
                   rir: np.ndarray) -> tuple[np.ndarray, dict]:
        """Add an Ambient recording bed BABBLE_BED_DB below a babble of speakers other than the `target` speaker."""
        talker_mix, streams = babble(rng, self.babble_speakers, target, n, rir)
        bed, details = self.recording_bed(rng, BABBLE_BED_RECORDINGS, n)
        ambient_gain = float(np.sqrt(np.mean(talker_mix ** 2) / np.mean(bed ** 2))
                             * 10 ** (BABBLE_BED_DB / 20))
        return talker_mix + ambient_gain * bed, {**details, "streams": streams,
                                                  "ambient_gain": ambient_gain}

    def noise_bed(self, rng: np.random.Generator, sample: Sample, n: int,
                  rir: np.ndarray) -> tuple[np.ndarray, dict]:
        """Build and reverberate the noise of `sample.noise_type`."""
        if sample.noise_type == "Babble":
            bed, details = self.babble_bed(rng, sample.speech_path, n, rir)
        else:
            bed, details = self.recording_bed(rng, sample.noise_type, n)
        return fftconvolve(bed, rir)[:n], details


def active_speech_power(speech: np.ndarray) -> float:
    """
    ITU-T P.56 active speech power (linear) via pyvoicebox `v_activlev`.

    Notes
    -----
    pyvoicebox-sap implements a simplified P.56: it skips the band-pass filtering of the full method.
    """
    power = float(v_activlev(speech, FS)[0])
    if not np.isfinite(power) or power <= 0:
        raise ValueError("P.56 active level is not positive; speech may be silent")
    return power


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float) -> tuple[np.ndarray, MixLevels]:
    """Mix noise at the P.56 speech SNR and return peak-normalised float32 audio."""
    speech = np.asarray(speech, dtype=np.float64)
    noise = np.asarray(noise, dtype=np.float64)
    speech_power = active_speech_power(speech)
    noise_power = float(np.mean(noise ** 2))
    gain = float(np.sqrt(speech_power / noise_power) * 10 ** (-snr_db / 20))  # puts the noise snr_db below the speech
    mixture = speech + gain * noise
    peak = float(np.max(np.abs(mixture)))
    if not np.isfinite(peak) or peak == 0:
        raise ValueError("speech/noise mixture is nonfinite or silent")
    return (mixture / peak).astype(np.float32), MixLevels(speech_power, noise_power, gain, peak)


def synthesize(sample: Sample, bank: NoiseBank) -> tuple[np.ndarray, dict]:
    """Reverberate speech and noise, then mix at the sample's SNR."""
    rir = load_audio(sample.rir_path)
    rir = rir / float(np.max(np.abs(rir)))
    reverberant = fftconvolve(load_audio(sample.speech_path), rir)
    n = len(reverberant)
    if n < MIN_SAMPLES:
        raise ValueError(f"reverberant speech shorter than {MIN_SAMPLES} samples")
    noise, details = bank.noise_bed(np.random.default_rng(sample.noise_seed), sample, n, rir)
    mixture, levels = mix_at_snr(reverberant, noise, sample.snr_db)
    details_json = json.dumps(details, sort_keys=True)
    return mixture, {**asdict(levels), "n_samples": n, "noise_details_json": details_json}


def write_sample(bank: NoiseBank, clips_dir: Path, sample: Sample) -> dict:
    """Save one FLOAT WAV and return its manifest row."""
    wav_name = f"{sample.sample_id}.wav"
    try:
        mixture, metadata = synthesize(sample, bank)
        # FLOAT keeps the samples exact for awssdr-decay-rates.
        sf.write(clips_dir / wav_name, mixture, FS, subtype="FLOAT")
    except Exception as exc:
        raise RuntimeError(f"sample {sample.sample_id}: {exc}") from exc
    return {**asdict(sample), **metadata, "wav_path": f"clips/{wav_name}"}


def _write_all(write_one: Callable[[Sample], dict], samples: list[Sample], jobs: int) -> list[dict]:
    if jobs == 1:
        return [write_one(sample) for sample in samples]
    with multiprocessing.get_context("spawn").Pool(jobs) as pool:
        # imap_unordered raises on the first failure; chunks keep the pickled NoiseBank to one copy per 64 samples
        return list(pool.imap_unordered(write_one, samples, chunksize=64))


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_dataset(output: Path, samples: list[Sample], bank: NoiseBank, *,
                  jobs: int = 1, options: dict | None = None) -> Path:
    """Write utterances to a new directory; leave partial output on failure."""
    output = output.absolute()
    # shortcut: incomplete output stays visible; use a new path or remove it before retrying.
    output.mkdir(parents=True)
    (output / "clips").mkdir()
    write_one = partial(write_sample, bank, output / "clips")
    rows = sorted(_write_all(write_one, samples, jobs), key=lambda row: row["sample_id"])
    _write_csv(output / "manifest.csv", rows)
    run = {"recipe_version": RECIPE_VERSION, "samples": len(rows), "options": options or {},
           "dependencies": {name: importlib.metadata.version(name) for name in DEPENDENCIES}}
    (output / "run.json").write_text(json.dumps(run, indent=2, default=str) + "\n")
    # Written last: awssdr-decay-rates only reads a dataset that has it.
    (output / "COMPLETE.json").write_text(
        json.dumps({"samples": len(rows), "recipe_version": RECIPE_VERSION}) + "\n")
    return output


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                        help="folder containing Speech, Noise and Eval_data (default: the DEFAULT_ROOT placeholder)")
    parser.add_argument("--rir-dir", type=Path, default=DEFAULT_RIR_DIR,
                        help="folder with the RIR .wav files and T30_ground_truth.csv "
                             "(default: the DEFAULT_RIR_DIR placeholder)")
    parser.add_argument("--out", type=Path,
                        help="new dataset directory (default: ROOT/training_utterances)")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit-rirs", type=int, default=0, help="0 means all RIRs")
    parser.add_argument("--sources", nargs="+", choices=SOURCES, default=list(SOURCES))
    parser.add_argument("--ears-passes", type=int, choices=range(1, PASSES.count("ears") + 1),
                        default=PASSES.count("ears"), help="number of EARS passes to keep (default all)")
    parser.add_argument("--noise-types", nargs="+", choices=NOISE_TYPES, default=list(NOISE_TYPES))
    parser.add_argument("--dry-run", action="store_true", help="plan and validate inputs; write nothing")
    args = parser.parse_args(argv)
    if args.jobs < 1 or args.seed < 0 or args.limit_rirs < 0:
        parser.error("--jobs must be positive; --seed and --limit-rirs nonnegative")
    return args


def main(argv: list[str] | None = None) -> None:
    """Plan and generate the dataset from command-line options; --dry-run only plans."""
    args = _parse_args(argv)
    root = args.root.resolve()
    rir_dir = args.rir_dir.resolve()
    output = args.out if args.out is not None else root / "training_utterances"
    rirs = load_rirs(rir_dir)[:args.limit_rirs or None]
    speech = {source: find_speech(root, source) for source in args.sources}
    samples = plan_samples(speech, rirs, args.seed, tuple(args.sources), tuple(args.noise_types),
                           args.ears_passes)
    bank = NoiseBank.load(root, tuple(args.noise_types))
    print(f"{RECIPE_VERSION}: {len(samples)} training utterances", flush=True)
    if args.dry_run:
        return
    options = {**vars(args), "root": root, "rir_dir": rir_dir, "out": output}
    print(f"Complete: {write_dataset(output, samples, bank, jobs=args.jobs, options=options)}")


if __name__ == "__main__":
    main()
