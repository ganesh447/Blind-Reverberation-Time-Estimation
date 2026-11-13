"""Combine RIR folders into one training pool, e.g. an RIR pool and its DRR copies from awssdr-drr.

Run: awssdr-pool --rir-dirs <POOL> <DRR_COPIES> --out <NEW_POOL>

<NEW_POOL> receives a symbolic link to every RIR listed in each folder's T30_ground_truth.csv, plus one
T30_ground_truth.csv with all of them and their labels: the layout awssdr-utterances reads.
File names must be unique across the folders.
"""

import argparse
import csv
from collections import Counter
from pathlib import Path

from awssdr.dataset_layout import RIR_LABELS_CSV


def read_labels(rir_dir: Path) -> list[tuple[Path, str]]:
    """(RIR path, T30 label) of every RIR listed in the folder's label CSV; each listed file must exist."""
    with (rir_dir / RIR_LABELS_CSV).open(newline="") as handle:
        labels = [(rir_dir / row["filename"], row["t30"]) for row in csv.DictReader(handle)]
    missing = [path.name for path, _ in labels if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{rir_dir}: listed in {RIR_LABELS_CSV} but missing: {', '.join(missing)}")
    return labels


def collect_pool(rir_dirs: list[Path]) -> list[tuple[Path, str]]:
    """All labelled RIRs of the folders, in folder order; their file names must be unique."""
    pool = [entry for rir_dir in rir_dirs for entry in read_labels(rir_dir)]
    counts = Counter(path.name for path, _ in pool)
    duplicates = sorted(name for name, count in counts.items() if count > 1)
    if duplicates:
        raise ValueError(f"file names listed more than once: {', '.join(duplicates)}")
    return pool


def write_pool(out: Path, pool: list[tuple[Path, str]]) -> None:
    out.mkdir(parents=True)  # refuses an existing folder, so no earlier pool is overwritten
    for path, _ in pool:
        (out / path.name).symlink_to(path.resolve())  # resolved, so links into another pool survive its removal
    with (out / RIR_LABELS_CSV).open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("filename", "t30"))
        writer.writerows((path.name, t30) for path, t30 in pool)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rir-dirs", type=Path, nargs="+", required=True,
                        help="folders with RIR .wav files and T30_ground_truth.csv")
    parser.add_argument("--out", type=Path, required=True, help="new pool folder")
    args = parser.parse_args(argv)
    pool = collect_pool(args.rir_dirs)  # every check runs before anything is written
    write_pool(args.out, pool)
    print(f"{len(pool)} RIRs from {len(args.rir_dirs)} folders: {args.out}")


if __name__ == "__main__":
    main()
