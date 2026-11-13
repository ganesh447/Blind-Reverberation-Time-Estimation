"""Checks for awssdr.acoustics.t30_labels."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from awssdr.dataset_layout import FS, RIR_LABELS_CSV
from awssdr.acoustics import t30_labels


def decaying_noise(t60: float, seconds: float = 2.0) -> np.ndarray:
    """Noise whose envelope falls 60 dB per t60 seconds over a -100 dB floor: an RIR with T30 = t60."""
    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(0)
    return rng.normal(size=len(t)) * 10 ** (-3 * t / t60) + 1e-5 * rng.normal(size=len(t))


class T30LabelsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.rir_dir = Path(directory.name)

    def write_rir(self, name: str, audio: np.ndarray) -> None:
        sf.write(self.rir_dir / name, audio, FS, subtype="FLOAT")

    def test_labels_match_known_decays(self):
        for t60 in (0.4, 0.8):
            self.write_rir(f"room_{t60}.wav", decaying_noise(t60))
        t30_labels.main(["--rir-dir", str(self.rir_dir)])
        with (self.rir_dir / RIR_LABELS_CSV).open() as handle:
            labels = {row["filename"]: float(row["t30"]) for row in csv.DictReader(handle)}
        self.assertEqual(set(labels), {"room_0.4.wav", "room_0.8.wav"})
        self.assertAlmostEqual(labels["room_0.4.wav"], 0.4, delta=0.02)
        self.assertAlmostEqual(labels["room_0.8.wav"], 0.8, delta=0.04)

    def test_all_failures_are_reported_and_nothing_is_written(self):
        self.write_rir("stereo.wav", np.stack([decaying_noise(0.5)] * 2, axis=1))
        self.write_rir("silent.wav", np.zeros(FS))
        with self.assertRaisesRegex(ValueError, r"2 RIR\(s\) failed(.|\n)*silent.wav(.|\n)*stereo.wav"):
            t30_labels.main(["--rir-dir", str(self.rir_dir)])
        self.assertFalse((self.rir_dir / RIR_LABELS_CSV).exists())

    def test_existing_labels_are_never_replaced(self):
        self.write_rir("room.wav", decaying_noise(0.5))
        (self.rir_dir / RIR_LABELS_CSV).write_text("filename,t30\nroom.wav,9.9\n")
        with self.assertRaises(FileExistsError):
            t30_labels.main(["--rir-dir", str(self.rir_dir)])
        self.assertIn("9.9", (self.rir_dir / RIR_LABELS_CSV).read_text())


if __name__ == "__main__":
    unittest.main()
