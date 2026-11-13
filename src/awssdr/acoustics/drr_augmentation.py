"""Make DRR-boosted copies of a pool's RIRs (after Bryan, ICASSP 2020, Sec. 3.2).

Run: awssdr-drr --rir-dir <POOL> --out <NEW_DIR> [--drr-lo 6 --drr-hi 16]

Each copy scales only the direct sound of its source RIR to a target DRR drawn from
U[max(drr_lo, source DRR + 1), drr_hi] dB. The reverberant tail is untouched, so a copy keeps its source's T30 label.
<NEW_DIR> receives:
  <stem>__drr<XXpX>.wav   the copies (16 kHz mono float32)
  T30_ground_truth.csv    their labels (filename, t30), taken from the source RIRs
  metadata.csv            one row per source RIR: windows, DRRs, gain, or why no copy was made

DRR definition:
  onset   first sample above -20 dB re the |h| peak
  direct  [onset - 0.5 ms, onset + 2.5 ms)
  late    [onset + 2.5 ms, Lundeby truncation), or to the end if the truncation lies within 60 ms of the onset
  DRR     10 log10(E_direct / E_late)
"""

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import NamedTuple

import numpy as np
import soundfile as sf

from awssdr.dataset_layout import FS, RIR_LABELS_CSV

ONSET_DB = -20.0
DIRECT_PRE, DIRECT_POST, EDGE = 8, 40, 4  # samples: 0.5 ms before the onset, 2.5 ms after, 0.25 ms cosine edges
MIN_LATE = int(0.06 * FS)
SECOND_ARRIVAL = (16, 40)  # samples after the peak (1-2.5 ms) that must hold no arrival within SECOND_ARRIVAL_DB
SECOND_ARRIVAL_DB = -6.0
MIN_BOOST_DB = 1.0
SEED = 0
METADATA_FIELDS = ("source_file", "t30", "onset", "trunc", "drr_orig", "filename", "drr_target", "drr_achieved",
                   "alpha", "alpha_db", "kept", "reason")


class Windows(NamedTuple):
    """Sample indices of one RIR: onset, direct part [direct_start, direct_end), late part [direct_end, late_end)."""
    onset: int
    direct_start: int
    direct_end: int
    late_end: int


def lundeby_truncation(h: np.ndarray, fs: int = FS, blk: float = 0.01, tail_skip: float = 0.05) -> int:
    """Sample where the decay meets the noise floor (simplified Lundeby); the late energy of the DRR ends here."""
    e = h.astype(np.float64) ** 2
    n = len(e) - int(tail_skip * fs)
    b = int(blk * fs)
    nb = n // b
    env = e[:nb * b].reshape(nb, b).mean(1)
    envdb = 10 * np.log10(env / env.max() + 1e-30)
    t = (np.arange(nb) + 0.5) * blk
    i0 = int(np.argmax(env))
    noise = 10 * np.log10(env[int(nb * 0.9):].mean() / env.max() + 1e-30)
    tc = t[-1]
    for _ in range(8):
        below = np.nonzero(envdb[i0:] < noise + 10)[0]
        stop = i0 + (below[0] if len(below) else nb - 1 - i0)
        if stop - i0 < 3:
            break
        k, c0 = np.polyfit(t[i0:stop], envdb[i0:stop], 1)
        if k >= 0:
            break
        tc_new = float(np.clip((noise - c0) / k, t[i0 + 1], t[-1]))
        j = min(int((tc_new + 10 / abs(k)) / blk), int(nb * 0.9))
        noise = 10 * np.log10(env[j:].mean() / env.max() + 1e-30)
        done = abs(tc_new - tc) < 0.002
        tc = tc_new
        if done:
            break
    return int(tc * fs)


def find_windows(h: np.ndarray) -> Windows:
    magnitude = np.abs(h)
    onset = int(np.argmax(magnitude > magnitude.max() * 10 ** (ONSET_DB / 20)))
    late_end = lundeby_truncation(h)
    if late_end <= onset + MIN_LATE:
        late_end = len(h)
    return Windows(onset, max(onset - DIRECT_PRE, 0), onset + DIRECT_POST, late_end)


def drr_db(h: np.ndarray, windows: Windows) -> float:
    energy = h.astype(np.float64) ** 2
    direct = energy[windows.direct_start:windows.direct_end].sum()
    return float(10 * np.log10(direct / energy[windows.direct_end:windows.late_end].sum()))


def skip_reason(h: np.ndarray, windows: Windows, source_drr: float, drr_hi: float) -> str | None:
    """Why this RIR gets no copy, or None if it can be boosted."""
    if source_drr >= drr_hi - MIN_BOOST_DB:
        return "source_drr_high"
    magnitude = np.abs(h)
    peak = int(np.argmax(magnitude))
    if peak >= windows.direct_end:
        # the first arrival is a weak (obstructed) path; boosting it would put a spike ahead of the main peak
        return "peak_after_direct_window"
    after_peak = magnitude[peak + SECOND_ARRIVAL[0]:peak + SECOND_ARRIVAL[1]]
    if after_peak.size and after_peak.max() >= magnitude[peak] * 10 ** (SECOND_ARRIVAL_DB / 20):
        # it would be boosted along with the direct sound
        return "strong_arrival_in_direct_window"
    return None


def direct_taper(length: int) -> np.ndarray:
    """Flat-top window over the direct part with EDGE-sample raised-cosine edges."""
    taper = np.ones(length)
    edge = 0.5 * (1 - np.cos(np.pi * (np.arange(EDGE) + 0.5) / EDGE))
    taper[:EDGE] = edge
    taper[-EDGE:] = edge[::-1]
    return taper


def boost_direct(h: np.ndarray, windows: Windows, target_db: float) -> tuple[np.ndarray, float]:
    """Copy of h whose direct part is scaled by (1 + beta * taper) to reach target_db; returns it and 1 + beta.

    beta >= 0 solves sum(h_d^2 (1 + beta w)^2) = 10^(target/10) E_late in closed form.
    """
    direct = h[windows.direct_start:windows.direct_end].astype(np.float64)
    taper = direct_taper(len(direct))
    late = float((h[windows.direct_end:windows.late_end].astype(np.float64) ** 2).sum())
    a, b, c = (direct ** 2).sum(), (taper * direct ** 2).sum(), (taper ** 2 * direct ** 2).sum()
    beta = (-b + np.sqrt(b * b - c * (a - 10 ** (target_db / 10) * late))) / c
    boosted = h.astype(np.float64).copy()
    boosted[windows.direct_start:windows.direct_end] = direct * (1 + beta * taper)
    return boosted.astype(np.float32), float(1 + beta)


def augment_rir(source_name: str, t30: str, h: np.ndarray, index: int, drr_lo: float,
                drr_hi: float) -> tuple[np.ndarray | None, dict]:
    """The boosted copy of one RIR (None if it gets no copy) and its complete metadata row."""
    windows = find_windows(h)
    source_drr = drr_db(h, windows)
    row = {"source_file": source_name, "t30": t30, "onset": windows.onset, "trunc": windows.late_end,
           "drr_orig": source_drr}
    reason = skip_reason(h, windows, source_drr, drr_hi)
    if reason:
        return None, {**row, "kept": False, "reason": reason}
    # shortcut: seeded by row number, so adding or removing a source RIR changes later targets; seed by file name
    # once existing pools no longer need to be rebuilt exactly
    target = float(np.random.default_rng([SEED, index]).uniform(max(drr_lo, source_drr + MIN_BOOST_DB), drr_hi))
    boosted, alpha = boost_direct(h, windows, target)
    return boosted, {**row, "filename": copy_name(source_name, target), "drr_target": target,
                     "drr_achieved": drr_db(boosted, windows), "alpha": alpha, "alpha_db": 20 * np.log10(alpha),
                     "kept": True, "reason": ""}


def read_sources(rir_dir: Path) -> list[tuple[str, str]]:
    """(file name, T30 label) of every RIR in the pool's T30_ground_truth.csv, in file order."""
    with (rir_dir / RIR_LABELS_CSV).open(newline="") as handle:
        return [(row["filename"], row["t30"]) for row in csv.DictReader(handle)]


def read_rir(path: Path) -> np.ndarray:
    h, sample_rate = sf.read(path, dtype="float32")
    if sample_rate != FS or h.ndim != 1:
        raise ValueError(f"{path.name}: expected a mono {FS} Hz RIR, got {sample_rate} Hz, shape {h.shape}")
    return h


def copy_name(source_name: str, target_db: float) -> str:
    drr_text = f"{target_db:04.1f}".replace(".", "p")
    return f"{Path(source_name).stem}__drr{drr_text}.wav"


def write_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_pool(out: Path, results: list[tuple[np.ndarray | None, dict]]) -> None:
    """Write the copies, their labels and the metadata of every source into the new folder `out`."""
    out.mkdir(parents=True)  # refuses an existing folder, so no earlier pool is overwritten
    for boosted, row in results:
        if boosted is not None:
            sf.write(out / row["filename"], boosted, FS, subtype="FLOAT")
    rows = [row for _, row in results]
    labels = [{"filename": row["filename"], "t30": row["t30"]} for row in rows if row["kept"]]
    write_csv(out / RIR_LABELS_CSV, ("filename", "t30"), labels)
    write_csv(out / "metadata.csv", METADATA_FIELDS, rows)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rir-dir", type=Path, required=True, help="pool with RIR .wav files and T30_ground_truth.csv")
    parser.add_argument("--out", type=Path, required=True, help="new folder for the copies")
    parser.add_argument("--drr-lo", type=float, default=6.0, help="lowest target DRR in dB")
    parser.add_argument("--drr-hi", type=float, default=16.0, help="highest target DRR in dB")
    args = parser.parse_args(argv)
    if not args.drr_lo < args.drr_hi:
        parser.error("need --drr-lo < --drr-hi")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    # every RIR is processed before anything is written, so bad input never leaves a partial folder
    results = [augment_rir(name, t30, read_rir(args.rir_dir / name), index, args.drr_lo, args.drr_hi)
               for index, (name, t30) in enumerate(read_sources(args.rir_dir))]
    write_pool(args.out, results)
    skipped = Counter(row["reason"] for _, row in results if not row["kept"])
    print(f"sources {len(results)}, copies {len(results) - sum(skipped.values())}, skipped {dict(skipped)}")


if __name__ == "__main__":
    main()
