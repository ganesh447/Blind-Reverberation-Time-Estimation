"""Checks for awssdr.training.train."""
import csv
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from awssdr.dataset_layout import DECAY_RATES_DIR, INDEX_FILE, LABELS_DIR, LEGACY_LABELS_DIR
from awssdr.training import train


def write_features(folder: Path, n_samples: int, seed: int) -> None:
    """A feature folder of random decay rates (40 mels, 30-60 frames) and labels."""
    rng = np.random.default_rng(seed)
    (folder / DECAY_RATES_DIR).mkdir(parents=True)
    (folder / LABELS_DIR).mkdir()
    index = []
    for i in range(n_samples):
        n_frames = int(rng.integers(30, 61))
        np.save(folder / DECAY_RATES_DIR / f"s{i}.npy", rng.normal(size=(40, n_frames)).astype(np.float32))
        np.save(folder / LABELS_DIR / f"s{i}.npy", np.float32(rng.uniform(0.2, 1.5)))
        index.append((f"s{i}", n_frames))
    with (folder / INDEX_FILE).open("wb") as handle:
        pickle.dump(index, handle)


class TrainTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        write_features(self.root / "train", 40, seed=1)
        write_features(self.root / "dev", 6, seed=2)

    def run_training(self, out: str) -> Path:
        with mock.patch("torch.cuda.is_available", return_value=False):  # CPU, so the run is bit-reproducible
            train.main(["--train", str(self.root / "train"), "--dev", str(self.root / "dev"),
                        "--out", str(self.root / out), "--epochs", "2", "--save-epochs-from", "2",
                        "--num-workers", "0"])
        return self.root / out

    def test_run_writes_weights_epochs_and_one_log_row_per_epoch(self):
        run = self.run_training("run")
        self.assertTrue((run / "latest_weights.pyt").is_file())
        self.assertEqual([p.name for p in (run / "epoch_checkpoints").iterdir()], ["epoch2_weights.pyt"])
        with (run / "train_log.csv").open() as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual([row["epoch"] for row in rows], ["1", "2"])
        self.assertEqual(rows[-1]["dev_n"], "6")

    def test_same_seed_gives_identical_weights(self):
        first = torch.load(self.run_training("first") / "latest_weights.pyt")
        second = torch.load(self.run_training("second") / "latest_weights.pyt")
        self.assertTrue(all(torch.equal(first[name], second[name]) for name in first))

    def test_dev_folder_with_the_old_t60_label_name_is_still_read(self):
        (self.root / "dev" / LABELS_DIR).rename(self.root / "dev" / LEGACY_LABELS_DIR)
        with (self.run_training("run") / "train_log.csv").open() as handle:
            self.assertEqual(list(csv.DictReader(handle))[-1]["dev_n"], "6")

    def test_existing_run_folder_is_refused(self):
        self.run_training("run")
        with self.assertRaises(FileExistsError):
            self.run_training("run")


if __name__ == "__main__":
    unittest.main()
