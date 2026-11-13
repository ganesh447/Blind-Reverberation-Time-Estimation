"""Checks for awssdr.training.average_checkpoints."""
import tempfile
import unittest
from pathlib import Path

import torch

from awssdr.training import average_checkpoints


class AverageCheckpointsTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.run = Path(directory.name)
        (self.run / "epoch_checkpoints").mkdir()
        for epoch, value in ((1, 1.0), (2, 3.0)):
            torch.save({"weight": torch.full((2,), value), "step": torch.tensor([epoch])},
                       self.run / "epoch_checkpoints" / f"epoch{epoch}_weights.pyt")

    def test_average_is_the_elementwise_mean(self):
        average_checkpoints.main(["--run", str(self.run), "--first", "1", "--last", "2"])
        average = torch.load(self.run / "swa_1_2_weights.pyt")
        torch.testing.assert_close(average["weight"], torch.full((2,), 2.0))
        torch.testing.assert_close(average["step"], torch.tensor([1.5]))

    def test_missing_epoch_is_named(self):
        with self.assertRaisesRegex(FileNotFoundError, "epoch3_weights.pyt"):
            average_checkpoints.main(["--run", str(self.run), "--first", "1", "--last", "3"])

    def test_inverted_range_is_rejected(self):
        with self.assertRaises(SystemExit):
            average_checkpoints.main(["--run", str(self.run), "--first", "2", "--last", "1"])


if __name__ == "__main__":
    unittest.main()
