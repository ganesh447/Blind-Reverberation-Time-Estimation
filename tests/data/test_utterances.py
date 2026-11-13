"""Checks for awssdr.data.utterances and its hand-off to awssdr.data.decay_rates."""
import csv
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import soundfile as sf

from awssdr.data import decay_rates
from awssdr.data import utterances


def by_speaker(*clips):
    """Clips grouped by speaker folder, the shape NoiseBank holds babble talkers in."""
    grouped = {}
    for clip in clips:
        grouped.setdefault(clip.parent, []).append(clip)
    return {folder: tuple(paths) for folder, paths in grouped.items()}


UNIT_IMPULSE = np.array([1.0])  # a room response that leaves the talkers unchanged


class MakeUtterancesTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="make-utterances-test-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        rng = np.random.default_rng(13)
        envelope = np.abs(np.sin(np.linspace(0, 6 * np.pi, 8000)))
        self.speech = (rng.normal(size=8000) * envelope).astype(np.float32)
        self.clip = self._wav("Speech/gsp/target.wav", self.speech)
        self.rir = self._wav("room.wav", np.exp(-np.arange(2048) / 300))
        recording = self._wav("ambient.wav", rng.normal(size=utterances.FS))
        event = rng.normal(size=512).astype(np.float32)
        self.bank = utterances.NoiseBank({"Ambient": (recording,)}, {},
                                  (utterances.Transient(event, float(np.sqrt(np.mean(event ** 2))), self.root / "e.wav"),))
        self.sample = utterances.Sample("test_sample", 3, "gsp", self.rir, self.clip, 0.45, "Ambient", 10, (7, 3, 0, 4))

    def _wav(self, name, audio):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, np.asarray(audio, dtype=np.float32), utterances.FS, subtype="FLOAT")
        return path

    def test_mixture_has_requested_p56_snr(self):
        noise = np.random.default_rng(1).normal(size=len(self.speech))
        speech_power = utterances.active_speech_power(self.speech)
        for snr_db in (0, 10, 20):
            mixture, levels = utterances.mix_at_snr(self.speech, noise, snr_db)
            scaled_noise = mixture * levels.mixture_peak - self.speech
            measured_db = 10 * np.log10(speech_power / np.mean(scaled_noise ** 2))
            self.assertAlmostEqual(measured_db, snr_db, delta=0.01)
            self.assertEqual(np.max(np.abs(mixture)), 1.0)

    def test_babble_uses_distinct_speakers_other_than_the_target(self):
        target = self._wav("Speech/EARS/p001/target.wav", self.speech)
        same_speaker = self._wav("Speech/EARS/p001/other.wav", self.speech)
        others = tuple(self._wav(f"Speech/EARS/p{i:03d}/clip.wav", self.speech) for i in range(2, 9))
        speakers = by_speaker(target, same_speaker, *others)
        _, streams = utterances.babble(np.random.default_rng(0), speakers, target, utterances.FS, UNIT_IMPULSE)
        self.assertTrue(4 <= len(streams) <= 7)
        self.assertEqual(len({stream["speaker"] for stream in streams}), len(streams))
        self.assertNotIn("p001", {stream["speaker"] for stream in streams})

    def test_babble_talkers_enter_the_mix_at_equal_levels(self):
        # Each talker is a tone of a different loudness; whole-cycle tones are orthogonal, so each one's
        # amplitude in the mix can be read off separately. Equal P.56 levels mean equal amplitudes.
        t = np.arange(utterances.FS) / utterances.FS
        tones = {f"p{i:03d}": np.sin(2 * np.pi * 100 * i * t) for i in range(2, 10)}
        target = self._wav("Speech/EARS/p001/target.wav", self.speech)
        talkers = tuple(self._wav(f"Speech/EARS/{speaker}/clip.wav", tone * int(speaker[1:]) / 10)
                        for speaker, tone in tones.items())
        mixture, streams = utterances.babble(np.random.default_rng(0), by_speaker(target, *talkers), target,
                                             utterances.FS, UNIT_IMPULSE)
        amplitudes = [2 * np.mean(mixture * tones[stream["speaker"]]) for stream in streams]
        np.testing.assert_allclose(amplitudes, amplitudes[0], rtol=1e-3)

    def test_restricted_plan_keeps_full_run_samples(self):
        speech = {source: tuple(self.root / "Speech" / source / f"{i}.wav" for i in range(5))
                  for source in utterances.SOURCES}
        rirs = [utterances.Rir(self.root / "a.wav", 0.4), utterances.Rir(self.root / "b.wav", 0.9)]
        full = utterances.plan_samples(speech, rirs, 42, utterances.SOURCES, utterances.NOISE_TYPES)
        subset = utterances.plan_samples(speech, rirs, 42, ("ears", "gsp"), ("Babble",))
        expected = [s for s in full if s.speech_source in ("ears", "gsp") and s.noise_type == "Babble"]
        self.assertEqual(subset, expected)
        self.assertEqual(len(full), len(utterances.PASSES) * len(rirs) * len(utterances.CONDITIONS))
        self.assertEqual(len({s.sample_id for s in full}), len(full))
        self.assertEqual([s.noise_seed for s in full[:2]], [(42, 0, 0, 0), (42, 0, 0, 1)])

    def test_fewer_ears_passes_keep_full_run_samples(self):
        speech = {source: tuple(self.root / "Speech" / source / f"{i}.wav" for i in range(5))
                  for source in utterances.SOURCES}
        rirs = [utterances.Rir(self.root / "a.wav", 0.4), utterances.Rir(self.root / "b.wav", 0.9)]
        full = utterances.plan_samples(speech, rirs, 42, utterances.SOURCES, utterances.NOISE_TYPES)
        one = utterances.plan_samples(speech, rirs, 42, utterances.SOURCES, utterances.NOISE_TYPES, ears_passes=1)
        self.assertEqual(one, [s for s in full if s.pass_index not in (1, 2)])

    def test_plan_without_samples_is_rejected(self):
        speech = {"gsp": (self.clip,)}
        with self.assertRaisesRegex(ValueError, "no samples"):
            utterances.plan_samples(speech, [], 42, ("gsp",), ("Ambient",))

    def test_babble_bed_draws_talkers_before_the_recording(self):
        others = tuple(self._wav(f"Speech/EARS/p00{i}/clip.wav", self.speech) for i in range(8))
        bank = utterances.NoiseBank(self.bank.recordings, by_speaker(*others), self.bank.transients)
        sample = replace(self.sample, noise_type="Babble", speech_path=others[0])
        seed = sample.noise_seed
        _, details = bank.noise_bed(np.random.default_rng(seed), sample, 8000, UNIT_IMPULSE)
        _, again = bank.noise_bed(np.random.default_rng(seed), sample, 8000, UNIT_IMPULSE)
        self.assertEqual(details, again)
        self.assertTrue(4 <= len(details["streams"]) <= 7)
        self.assertEqual(len({stream["speaker"] for stream in details["streams"]}), len(details["streams"]))

    def test_ears_uses_every_speaker_folder_it_finds(self):
        for speaker in ("p001", "p002", "p107"):
            self._wav(f"Speech/EARS/{speaker}/rainbow_1.wav", self.speech)
        self._wav("Speech/EARS/p107/other_take.wav", self.speech)
        clips = utterances.find_speech(self.root, "ears")
        self.assertEqual({clip.parent.name for clip in clips}, {"p001", "p002", "p107"})
        self.assertEqual(len(clips), 3)

    def test_missing_noise_recording_fails_before_generation(self):
        with self.assertRaisesRegex(FileNotFoundError, "AirConditioner_1"):
            utterances.checked_recording_paths(self.root, ("Fan",))

    def test_babble_checks_the_recordings_its_bed_is_shaped_from(self):
        with mock.patch.object(utterances, "BABBLE_BED_RECORDINGS", "Fan"):
            with self.assertRaisesRegex(FileNotFoundError, "AirConditioner_1"):
                utterances.checked_recording_paths(self.root, ("Babble",))

    def test_dataset_is_float_wav_readable_by_decay_rates(self):
        expected, _ = utterances.synthesize(self.sample, self.bank)
        out = utterances.write_dataset(self.root / "dataset", [self.sample], self.bank)
        waveform, sample_rate = sf.read(out / "clips/test_sample.wav", dtype="float32")
        np.testing.assert_array_equal(waveform, expected)
        self.assertEqual((sample_rate, sf.info(out / "clips/test_sample.wav").subtype), (utterances.FS, "FLOAT"))
        with (out / "manifest.csv").open() as handle:
            row = next(csv.DictReader(handle))
        self.assertEqual((row["n_samples"], float(row["target_t30"])), (str(len(waveform)), 0.45))
        self.assertIn("transients", json.loads(row["noise_details_json"]))
        self.assertEqual(json.loads((out / "COMPLETE.json").read_text())["samples"], 1)
        features = decay_rates.compute_dataset(out, self.root / "features", front_ends=("4k",))
        self.assertTrue((features / decay_rates.feature_dir("4k") / "decayRates" / "test_sample.npy").is_file())

    def test_failed_run_is_left_incomplete_and_retries_need_a_new_folder(self):
        out = self.root / "dataset"
        broken = replace(self.sample, sample_id="broken", speech_path=self.root / "missing.wav")
        with self.assertRaisesRegex(RuntimeError, "broken"):
            utterances.write_dataset(out, [self.sample, broken], self.bank, jobs=2)
        self.assertFalse((out / "COMPLETE.json").exists())
        with self.assertRaises(FileExistsError):
            utterances.write_dataset(out, [self.sample], self.bank)
        self.assertTrue((utterances.write_dataset(self.root / "retry", [self.sample], self.bank) / "COMPLETE.json").is_file())

    def test_competing_run_is_refused_before_generation_and_the_dataset_survives(self):
        out = self.root / "dataset"
        generate = utterances._write_all

        def generate_while_competitor_starts(*args):
            with self.assertRaises(FileExistsError):
                utterances.write_dataset(out, [self.sample], self.bank)
            return generate(*args)

        with mock.patch.object(utterances, "_write_all", generate_while_competitor_starts):
            utterances.write_dataset(out, [self.sample], self.bank)
        self.assertTrue((out / "COMPLETE.json").is_file())

    def test_dataset_directory_has_the_default_permissions(self):
        reference = self.root / "reference"
        reference.mkdir()
        out = utterances.write_dataset(self.root / "dataset", [self.sample], self.bank)
        self.assertEqual(out.stat().st_mode, reference.stat().st_mode)

    def test_run_json_records_the_recipe_and_dependency_versions(self):
        out = utterances.write_dataset(self.root / "dataset", [self.sample], self.bank)
        run = json.loads((out / "run.json").read_text())
        self.assertEqual(run["recipe_version"], utterances.RECIPE_VERSION)
        self.assertEqual(set(run["dependencies"]), set(utterances.DEPENDENCIES))


class CommandLineTests(unittest.TestCase):
    """Runs main() on a miniature copy of the T60_EST6 data layout."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="make-utterances-cli-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.rng = np.random.default_rng(5)
        self.build_data_tree()
        self.arguments = ["--root", str(self.root), "--rir-dir", str(self.root / "rirs"), "--seed", "7"]

    def write_noise_wav(self, relative_path, samples):
        path = self.root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(path, self.rng.normal(size=samples).astype(np.float32), utterances.FS, subtype="FLOAT")

    def build_data_tree(self):
        for speaker in range(10):
            self.write_noise_wav(f"Speech/EARS/p{speaker:03d}/rainbow_01_regular.wav", utterances.FS)
        self.write_noise_wav("Speech/SET-2/GSP/word/a.wav", 8000)
        for held_out_list in ("validation_list.txt", "testing_list.txt"):
            (self.root / "Speech/SET-2/GSP" / held_out_list).write_text("")
        for env in utterances.DEMAND_ENVS:
            self.write_noise_wav(f"Noise/DEMAND/{env}_16k/{env}/ch01.wav", utterances.FS)
        for transient_class in utterances.TRANSIENT_CLASSES:
            self.write_noise_wav(f"Eval_data/AID/wavs/{transient_class}_01_SH_MKH800.wav", 512)
        for name in ("a.wav", "b.wav"):
            self.write_noise_wav(f"rirs/{name}", 2048)
        (self.root / "rirs/T30_ground_truth.csv").write_text("filename,t30\na.wav,0.4\nb.wav,0.9\n")

    def run_main(self, *options):
        with redirect_stdout(io.StringIO()) as printed:
            utterances.main(self.arguments + list(options))
        return printed.getvalue()

    def test_unset_input_folders_fail_naming_the_placeholder(self):
        with self.assertRaisesRegex(FileNotFoundError, "PATH/TO/YOUR/RIR_FOLDER"):
            utterances.main(["--out", str(self.root / "dataset"), "--dry-run"])

    def test_dry_run_plans_the_limited_selection_and_writes_nothing(self):
        out = self.root / "dry"
        printed = self.run_main("--out", str(out), "--sources", "gsp", "--noise-types", "Ambient",
                                "--limit-rirs", "1", "--dry-run")
        self.assertIn("3 training utterances", printed)
        self.assertFalse(out.exists())

    def test_babble_run_finds_its_own_ears_clips(self):
        out = self.root / "dataset"
        self.run_main("--out", str(out), "--sources", "gsp", "--noise-types", "Babble", "--limit-rirs", "1")
        with (out / "manifest.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        streams = json.loads(rows[0]["noise_details_json"])["streams"]
        self.assertTrue(all("/EARS/" in stream["path"] for stream in streams))
        self.assertTrue(all(stream["path"].endswith("_regular.wav") for stream in streams))

    def test_babble_pool_keeps_plain_speech_and_excludes_stylized_readings(self):
        self.write_noise_wav("Speech/EARS/p000/rainbow_01_whisper.wav", utterances.FS)
        self.write_noise_wav("Speech/EARS/p000/freeform_speech_01.wav", utterances.FS)
        speakers = utterances.load_babble_speakers(self.root)
        names = {clip.name for clips in speakers.values() for clip in clips}
        self.assertEqual(len(speakers), 10)
        self.assertEqual(speakers[self.root / "Speech/EARS/p000"][0].name, "freeform_speech_01.wav")
        self.assertNotIn("rainbow_01_whisper.wav", names)

    def test_babble_speakers_keep_every_clip_whatever_the_listing_order(self):
        self.write_noise_wav("Speech/EARS/p000/freeform_speech_01.wav", utterances.FS)
        clips = utterances.find_speech(self.root, "ears")
        listing = clips[1:] + clips[:1]  # p000's first clip moved to the end, splitting p000's clips apart
        with mock.patch.object(utterances, "find_speech", return_value=listing):
            speakers = utterances.load_babble_speakers(self.root)
        self.assertEqual(sum(len(clips) for clips in speakers.values()), 11)
        self.assertEqual(len(speakers[self.root / "Speech/EARS/p000"]), 2)

    def test_dry_run_rejects_too_few_babble_speakers(self):
        for speaker in range(utterances.STREAM_COUNT[1], 10):
            shutil.rmtree(self.root / f"Speech/EARS/p{speaker:03d}")
        with self.assertRaisesRegex(ValueError, f"needs {utterances.STREAM_COUNT[1] + 1} EARS speakers"):
            self.run_main("--out", str(self.root / "dry"), "--sources", "gsp", "--noise-types", "Babble", "--dry-run")


if __name__ == "__main__":
    unittest.main()
