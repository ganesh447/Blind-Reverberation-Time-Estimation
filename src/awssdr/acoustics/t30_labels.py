"""Write the T30 label of every RIR in a folder, for awssdr-utterances.

Run: awssdr-t30-labels --rir-dir <RIR_DIR>

Measures the broadband Lundeby T30 of each mono *.wav directly in RIR_DIR and writes
RIR_DIR/T30_ground_truth.csv with the columns filename and t30 (seconds).
"""

import argparse
import csv
import math
from pathlib import Path

import soundfile as sf

from awssdr.dataset_layout import RIR_LABELS_CSV
from awssdr.acoustics.acoustic_features import calculate_reverberation_times


def measure_t30(rir_path: Path) -> float:
    """Lundeby T30 of one mono RIR in seconds."""
    ir, sample_rate = sf.read(rir_path)
    if ir.ndim != 1:
        raise ValueError("not mono; awssdr-utterances would mix its channels while this measures one")
    t30 = float(calculate_reverberation_times(ir, sample_rate, metrics="T30", method="lundeby")["T30"])
    if not (math.isfinite(t30) and t30 > 0):
        raise ValueError(f"invalid T30: {t30}")
    return t30


def measure_folder(rir_dir: Path) -> dict[str, float]:
    """T30 per RIR file name; reports every failing RIR at once."""
    labels, failures = {}, []
    for rir_path in sorted(rir_dir.glob("*.wav")):
        try:
            labels[rir_path.name] = measure_t30(rir_path)
        except Exception as exc:
            failures.append(f"{rir_path.name}: {exc}")
    if failures:
        raise ValueError(f"{len(failures)} RIR(s) failed, no labels written:\n" + "\n".join(failures))
    if not labels:
        raise FileNotFoundError(f"no *.wav files in {rir_dir}")
    return labels


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rir-dir", type=Path, required=True, help="folder of RIR .wav files")
    args = parser.parse_args(argv)
    labels = measure_folder(args.rir_dir)
    output = args.rir_dir / RIR_LABELS_CSV
    with output.open("x", newline="") as handle:  # "x": existing labels define existing datasets, never replace them
        writer = csv.writer(handle)
        writer.writerow(("filename", "t30"))
        writer.writerows(labels.items())
    print(f"wrote {len(labels)} T30 labels: {output}")


if __name__ == "__main__":
    main()
