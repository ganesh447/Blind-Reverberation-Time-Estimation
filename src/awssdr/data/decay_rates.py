"""Compute author-compatible SDRs from saved training-utterance WAVs.

Run: awssdr-decay-rates --utterances /path/to/corpora/training_utterances --out training_sdrs
"""

import argparse
import csv
import importlib.metadata
import io
import json
import multiprocessing
import pickle
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from awssdr.dataset_layout import DECAY_RATES_DIR, FS, INDEX_FILE, LABELS_DIR

N_MELS, N_FFT, WIN, HOP = 40, 512, 256, 128
FIT_LEN, FIT_HOP = 20, 2
MIN_SAMPLES = N_FFT + (FIT_LEN - 1) * HOP  # shortest input with one SDR fit window
FRONT_ENDS = {"6k": 6000.0, "4k": 4000.0}
FEATURE_VERSION = "author-legacy-mel-centres-v1"


def feature_dir(front_end: str) -> str:
    """Folder name the trainer is pointed at for one front end."""
    return f"v1-{front_end}"

_times = (np.linspace(1, FIT_LEN, FIT_LEN) / FS * WIN).reshape(-1, 1)
_SLOPE = np.linalg.pinv(np.concatenate((np.ones_like(_times), _times), axis=1))[1]


def mel_basis(front_end: str) -> np.ndarray:
    """Mel filterbank with 40 centres from 400 Hz to the front end's upper frequency."""
    hi = FRONT_ENDS[front_end]
    lo_mel, hi_mel = librosa.hz_to_mel(400.0), librosa.hz_to_mel(hi)
    step = (hi_mel - lo_mel) / (N_MELS - 1)
    # librosa takes the outer band edges, one centre spacing beyond the first and last centre
    return librosa.filters.mel(sr=FS, n_fft=N_FFT, n_mels=N_MELS,
                               fmin=float(librosa.mel_to_hz(lo_mel - step)),
                               fmax=float(librosa.mel_to_hz(hi_mel + step)))


def _validate_front_ends(front_ends: tuple[str, ...]) -> None:
    """Require a nonempty selection of distinct, known front-end names."""
    if not front_ends or len(set(front_ends)) != len(front_ends):
        raise ValueError("unique, nonempty front ends are required")
    if any(name not in FRONT_ENDS for name in front_ends):
        raise ValueError("unknown front end")


def decay_rates(mixture: np.ndarray, front_ends: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Map each front end to its float32 SDR array of shape (40, n_decay_frames), sharing one STFT."""
    _validate_front_ends(front_ends)
    if mixture.ndim != 1 or not mixture.size or not np.isfinite(mixture).all():
        raise ValueError("feature input: expected nonempty finite mono audio")
    if not np.any(mixture):
        raise ValueError("feature input: silent audio")
    if len(mixture) < MIN_SAMPLES:
        raise ValueError(f"feature input shorter than {MIN_SAMPLES} samples")

    magnitude = np.abs(librosa.stft(y=mixture, n_fft=N_FFT, hop_length=HOP,
                                   win_length=WIN, window="hamming", center=False))

    features = {}
    for name in front_ends:
        mel = mel_basis(name) @ magnitude
        # the legacy log has no floor, so zero energy is rejected instead of written as -inf/NaN features
        if not np.isfinite(mel).all() or np.any(mel <= 0):
            raise ValueError(f"{name}: nonpositive/nonfinite Mel energy; refusing invalid SDRs")
        windows = np.lib.stride_tricks.sliding_window_view(np.log(mel), FIT_LEN, axis=1)[:, ::FIT_HOP]
        rates = (windows @ _SLOPE).astype(np.float32)
        if not np.isfinite(rates).all():
            raise ValueError(f"{name}: nonfinite SDRs")
        features[name] = rates
    return features


def _save_sample(row: dict, *, utterances: Path, output: Path,
                 front_ends: tuple[str, ...]) -> dict:
    """Read one saved WAV and write its SDR arrays and training labels."""
    sample_id = row["sample_id"]
    try:
        wav_path = (utterances / row["wav_path"]).resolve()
        if not wav_path.is_relative_to(utterances):
            raise ValueError("WAV path leaves the utterance directory")
        waveform, sample_rate = sf.read(wav_path, dtype="float32")
        if sample_rate != FS or waveform.ndim != 1:
            raise ValueError("expected mono WAV audio at 16000 Hz; no resampling is performed")
        if len(waveform) != int(row["n_samples"]):
            raise ValueError("WAV length differs from the utterance manifest")
        features = decay_rates(waveform, front_ends)
        result = {**row, "n_decay_frames": next(iter(features.values())).shape[1]}
        for name, rates in features.items():
            directory = Path(feature_dir(name))
            feature_path = directory / DECAY_RATES_DIR / f"{sample_id}.npy"
            np.save(output / feature_path, rates, allow_pickle=False)
            np.save(output / directory / LABELS_DIR / f"{sample_id}.npy",
                    np.float32(row["target_t30"]), allow_pickle=False)
            result[f"sdr_path_{name}"] = str(feature_path)
        return result
    except Exception as exc:
        raise RuntimeError(f"sample {sample_id}: {exc}") from exc


def read_utterances(utterances: Path) -> tuple[list[dict], dict]:
    """Return the manifest rows and COMPLETE.json of a finished utterance folder."""
    completion = json.loads((utterances / "COMPLETE.json").read_text())
    if (not isinstance(completion, dict)
            or type(completion.get("samples")) is not int or completion["samples"] <= 0
            or not isinstance(completion.get("recipe_version"), str)
            or not completion["recipe_version"].strip()):
        raise ValueError("COMPLETE.json requires a positive integer samples count and a nonempty recipe_version")
    reader = csv.DictReader(io.StringIO((utterances / "manifest.csv").read_text(encoding="utf-8")))
    required = {"sample_id", "wav_path", "n_samples", "target_t30"}
    missing = required - set(reader.fieldnames or ())
    if missing:
        raise ValueError(f"utterance manifest missing columns: {', '.join(sorted(missing))}")
    rows = list(reader)
    if any(not row.get(name) for row in rows for name in required):
        raise ValueError("utterance manifest has empty required fields")
    ids = [row["sample_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids) or len(rows) != completion["samples"]:
        raise ValueError("utterance manifest must have unique IDs and match the completion count")
    # labels go straight into training; a bad n_samples is caught per sample against the WAV length
    if not all(0 < float(row["target_t30"]) < np.inf for row in rows):
        raise ValueError("T30 labels must be finite and positive")
    return rows, completion


def compute_dataset(utterances: Path, output: Path, *,
                    front_ends: tuple[str, ...] = ("6k", "4k"), jobs: int = 1) -> Path:
    """Write SDRs and T30 labels for a finished utterance folder into a new `output` directory."""
    utterances = utterances.resolve()
    output = output.absolute()
    if jobs < 1:
        raise ValueError("jobs must be positive")
    _validate_front_ends(front_ends)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to replace dataset: {output}")
    rows, completion = read_utterances(utterances)
    output.mkdir(parents=True)
    for name in front_ends:
        (output / feature_dir(name) / DECAY_RATES_DIR).mkdir(parents=True)
        (output / feature_dir(name) / LABELS_DIR).mkdir()
    save = partial(_save_sample, utterances=utterances, output=output, front_ends=front_ends)
    if jobs == 1:
        results = [save(row) for row in rows]
    else:
        with multiprocessing.get_context("spawn").Pool(jobs) as pool:
            results = list(pool.imap_unordered(save, rows, chunksize=1))
    results.sort(key=lambda row: row["sample_id"])
    with (output / "manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    index = [(row["sample_id"], row["n_decay_frames"]) for row in results]
    for name in front_ends:
        with (output / feature_dir(name) / INDEX_FILE).open("wb") as handle:
            pickle.dump(index, handle, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "artifact_type": "spectral_decay_rates", "feature_version": FEATURE_VERSION,
        "recipe_version": completion["recipe_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "samples": len(results), "front_ends": front_ends, "jobs": jobs,
        "signal_processing": {
            "sample_rate": FS, "n_fft": N_FFT, "window": "hamming",
            "win_length": WIN, "hop_length": HOP, "center": False,
            "n_mels": N_MELS, "mel_centres_lo_hz": 400,
            "mel_centres_hi_hz": {name: FRONT_ENDS[name] for name in front_ends},
            "mel_value": "magnitude", "log": "natural, no floor; reject nonpositive energy",
            "fit_length": FIT_LEN, "fit_hop": FIT_HOP,
            "slope_time_step_samples": WIN,
        },
        "dependencies": {name: importlib.metadata.version(name)
                         for name in ("numpy", "scipy", "librosa", "soundfile")},
    }
    (output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def main(argv=None) -> None:
    """Compute SDRs from the command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--utterances", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--front-ends", nargs="+", choices=tuple(FRONT_ENDS), default=list(FRONT_ENDS))
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args(argv)
    if args.jobs < 1:
        parser.error("jobs must be positive")
    output = compute_dataset(args.utterances, args.out, front_ends=tuple(args.front_ends), jobs=args.jobs)
    print(f"Complete: {output}")


if __name__ == "__main__":
    main()
