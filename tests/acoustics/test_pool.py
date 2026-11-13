"""Checks for awssdr.acoustics.pool."""
import tempfile
import unittest
from pathlib import Path

from awssdr.data.utterances import load_rirs
from awssdr.dataset_layout import RIR_LABELS_CSV
from awssdr.acoustics import pool


class PoolTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def rir_folder(self, name: str, labels: dict[str, str]) -> Path:
        folder = self.root / name
        folder.mkdir()
        for filename in labels:
            (folder / filename).write_bytes(b"RIFF")
        rows = "".join(f"{filename},{t30}\n" for filename, t30 in labels.items())
        (folder / RIR_LABELS_CSV).write_text("filename,t30\n" + rows)
        return folder

    def test_pool_links_every_rir_and_is_readable_by_the_utterance_step(self):
        rirs = self.rir_folder("rirs", {"a.wav": "0.4", "b.wav": "0.9"})
        copies = self.rir_folder("copies", {"a__drr12p0.wav": "0.4"})
        pool.main(["--rir-dirs", str(rirs), str(copies), "--out", str(self.root / "pool")])
        loaded = {rir.path.name: rir.t30 for rir in load_rirs(self.root / "pool")}
        self.assertEqual(loaded, {"a.wav": 0.4, "b.wav": 0.9, "a__drr12p0.wav": 0.4})
        self.assertEqual((self.root / "pool" / "a.wav").resolve(), (rirs / "a.wav").resolve())

    def test_duplicate_file_names_are_rejected_before_any_output(self):
        first = self.rir_folder("first", {"a.wav": "0.4"})
        second = self.rir_folder("second", {"a.wav": "0.5"})
        with self.assertRaisesRegex(ValueError, "a.wav"):
            pool.main(["--rir-dirs", str(first), str(second), "--out", str(self.root / "pool")])
        self.assertFalse((self.root / "pool").exists())

    def test_listed_but_missing_rir_is_named(self):
        rirs = self.rir_folder("rirs", {"a.wav": "0.4"})
        (rirs / "a.wav").unlink()
        with self.assertRaisesRegex(FileNotFoundError, "a.wav"):
            pool.main(["--rir-dirs", str(rirs), "--out", str(self.root / "pool")])


if __name__ == "__main__":
    unittest.main()
