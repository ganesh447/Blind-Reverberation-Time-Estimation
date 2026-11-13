"""Average a training run's epoch checkpoints into one model (stochastic weight averaging).

Run: awssdr-average --run <RUN_DIR> --first 90 --last 100

Reads <RUN_DIR>/epoch_checkpoints/epoch<N>_weights.pyt, as written by awssdr-train, and writes
<RUN_DIR>/swa_<first>_<last>_weights.pyt. RT_est has no batch normalisation, so the plain mean of the
weights is the complete model: no statistics need recomputing afterwards.
"""

import argparse
from pathlib import Path

import torch


def epoch_weight_paths(run: Path, first: int, last: int) -> list[Path]:
    """Weight files of epochs first..last; all of them must exist."""
    paths = [run / "epoch_checkpoints" / f"epoch{epoch}_weights.pyt" for epoch in range(first, last + 1)]
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing in {run / 'epoch_checkpoints'}: {', '.join(missing)} "
                                "(awssdr-train keeps epochs from --save-epochs-from on)")
    return paths


def average_weights(paths: list[Path]) -> dict[str, torch.Tensor]:
    """Elementwise mean of every tensor across the weight files."""
    state_dicts = [torch.load(path, map_location="cpu") for path in paths]
    names = state_dicts[0].keys()
    for path, state_dict in zip(paths, state_dicts):
        if state_dict.keys() != names:
            raise ValueError(f"{path.name} holds different tensors than {paths[0].name}")
    # float() also turns the integer step counter into a float, as earlier averages did; loading casts it back
    return {name: torch.stack([state_dict[name].float() for state_dict in state_dicts]).mean(dim=0)
            for name in names}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="awssdr-train output folder")
    parser.add_argument("--first", type=int, default=90, help="first epoch to average")
    parser.add_argument("--last", type=int, default=100, help="last epoch to average, inclusive")
    args = parser.parse_args(argv)
    if not 1 <= args.first <= args.last:
        parser.error("need 1 <= --first <= --last")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output = args.run / f"swa_{args.first}_{args.last}_weights.pyt"
    torch.save(average_weights(epoch_weight_paths(args.run, args.first, args.last)), output)
    print(f"averaged epochs {args.first}-{args.last}: {output}")


if __name__ == "__main__":
    main()
