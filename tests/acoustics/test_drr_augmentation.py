"""Checks for awssdr.acoustics.drr_augmentation."""
import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from awssdr.dataset_layout import FS, RIR_LABELS_CSV
from awssdr.acoustics import drr_augmentation

ONSET = 160


def rir_with_direct_sound(tail_level: float = 0.05) -> np.ndarray:
    """A direct impulse at ONSET followed by an exponentially decaying noise tail (T60 0.5 s)."""
    h = np.zeros(FS, dtype=np.float32)
    h[ONSET] = 1.0
    t = np.arange(FS - ONSET - 48) / FS
    h[ONSET + 48:] = tail_level * np.random.default_rng(0).normal(size=len(t)) * 10 ** (-3 * t / 0.5)
    return h


class DrrAugmentationTests(unittest.TestCase):
    def test_copy_reaches_the_target_and_keeps_the_tail(self):
        h = rir_with_direct_sound()
        boosted, row = drr_augmentation.augment_rir("room.wav", "0.5", h, 0, 6.0, 16.0)
        self.assertTrue(row["kept"])
        self.assertTrue(6.0 <= row["drr_target"] <= 16.0)
        self.assertAlmostEqual(row["drr_achieved"], row["drr_target"], delta=1e-3)
        direct = slice(ONSET - drr_augmentation.DIRECT_PRE, ONSET + drr_augmentation.DIRECT_POST)
        untouched = np.ones(len(h), dtype=bool)
        untouched[direct] = False
        np.testing.assert_array_equal(boosted[untouched], h[untouched])

    def test_rir_whose_strongest_arrival_follows_the_direct_window_is_skipped(self):
        h = rir_with_direct_sound()
        h[ONSET + 100] = 2.0
        boosted, row = drr_augmentation.augment_rir("room.wav", "0.5", h, 0, 6.0, 16.0)
        self.assertIsNone(boosted)
        self.assertEqual(row["reason"], "peak_after_direct_window")

    def test_pool_gets_only_copies_with_their_source_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            pool, out = Path(directory) / "pool", Path(directory) / "copies"
            pool.mkdir()
            sf.write(pool / "room.wav", rir_with_direct_sound(), FS, subtype="FLOAT")
            (pool / RIR_LABELS_CSV).write_text("filename,t30\nroom.wav,0.5\n")
            drr_augmentation.main(["--rir-dir", str(pool), "--out", str(out)])
            with (out / RIR_LABELS_CSV).open() as handle:
                labels = list(csv.DictReader(handle))
            self.assertEqual(sorted(p.name for p in out.glob("*.wav")), [labels[0]["filename"]])
            self.assertTrue(labels[0]["filename"].startswith("room__drr"))
            self.assertEqual(labels[0]["t30"], "0.5")

    def test_inverted_drr_range_is_rejected_before_any_output(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "copies"
            with self.assertRaises(SystemExit):
                drr_augmentation.main(["--rir-dir", directory, "--out", str(out), "--drr-lo", "16", "--drr-hi", "10"])
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
