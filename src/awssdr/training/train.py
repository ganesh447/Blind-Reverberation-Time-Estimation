"""Train the AWSSDR T30 estimator on decay-rate features.

Run: awssdr-train --train <FEATURES>/v1-6k --dev <DEV_FEATURES> --out <RUN_DIR>

Both feature folders hold decayRates/, T30/ and dataset.pkl, as written by awssdr-decay-rates.
awssdr-average then averages the saved epoch checkpoints into the delivered model.
"""

import argparse
import csv
import json
import pickle
import time
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from awssdr.dataset_layout import DECAY_RATES_DIR, INDEX_FILE, LABELS_DIR, LEGACY_LABELS_DIR
from awssdr.models.rt_est import RT_est

N_MELS = 40
LOG_EVERY = 25
LOG_FIELDS = ("epoch", "train_loss", "dev_bias", "dev_mae", "dev_mse", "dev_pearson_rho", "dev_n")


def read_index(feature_dir: Path) -> list[tuple[str, int]]:
    """(sample_id, n_frames) pairs from dataset.pkl."""
    with (feature_dir / INDEX_FILE).open("rb") as handle:
        return pickle.load(handle)


def labels_dir(feature_dir: Path) -> Path:
    """The label folder of a feature folder; folders made before the T30 rename call it T60."""
    # shortcut: reads the old name so the current ACE dev sets keep working; drop once they are rebuilt
    # with awssdr-decay-rates
    labels = feature_dir / LABELS_DIR
    return labels if labels.is_dir() else feature_dir / LEGACY_LABELS_DIR


def load_sample(feature_dir: Path, sample_id: str) -> tuple[np.ndarray, np.ndarray]:
    """Decay rates (mels, frames) and T30 label of one sample."""
    return (np.load(feature_dir / DECAY_RATES_DIR / f"{sample_id}.npy"),
            np.load(labels_dir(feature_dir) / f"{sample_id}.npy"))


def load_batch(feature_dir: Path, sample_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero-pad a batch's decay rates to (batch, frames, mels) and stack its labels."""
    samples = [load_sample(feature_dir, sample_id) for sample_id in sample_ids]
    rates = torch.nn.utils.rnn.pad_sequence([torch.from_numpy(dr.T) for dr, _ in samples], batch_first=True)
    labels = torch.stack([torch.as_tensor(t30, dtype=torch.float32) for _, t30 in samples])
    return rates, labels


def length_batches(feature_dir: Path, batch_size: int) -> list[list[str]]:
    """Cut the samples, sorted longest first, into consecutive batches of similar length."""
    index = sorted(read_index(feature_dir), key=lambda entry: (int(entry[1]), entry[0]), reverse=True)
    longest_first = [sample_id for sample_id, _ in index]
    return [longest_first[i:i + batch_size] for i in range(0, len(longest_first), batch_size)]


def epoch_loader(feature_dir: Path, batches: list[list[str]], order: torch.Generator,
                 num_workers: int) -> DataLoader:
    """Load the batches in a new order drawn from `order`."""
    shuffled = [batches[i] for i in torch.randperm(len(batches), generator=order).tolist()]
    return DataLoader(shuffled, batch_size=None, collate_fn=partial(load_batch, feature_dir),
                      num_workers=num_workers, pin_memory=torch.cuda.is_available())


def init_weights(module: torch.nn.Module) -> None:
    """Kaiming-normal weights and zero biases for the conv and linear layers."""
    if isinstance(module, (torch.nn.Conv1d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(module.weight.data)
        module.bias.data.fill_(0)


def apply_gradients(model: RT_est, optimizer: torch.optim.Optimizer, grad_clip: float) -> None:
    """Clip the summed gradients, take one optimizer step and clear them."""
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if torch.isnan(grad_norm):
        raise FloatingPointError("gradient norm is NaN")
    optimizer.step()
    optimizer.zero_grad()


def train_epoch(model: RT_est, optimizer: torch.optim.Optimizer, loader: DataLoader,
                update_period: int, grad_clip: float, device: torch.device) -> float:
    """One pass over `loader`, summing the gradients of update_period batches per step; returns the mean loss."""
    model.train()
    running_loss, pending, start = 0.0, 0, time.time()
    for batch_index, (rates, labels) in enumerate(loader, 1):
        prediction, *_ = model(rates.to(device, torch.float32))
        loss = torch.nn.functional.mse_loss(prediction, labels.to(device, torch.float32).view(-1, 1))
        loss.backward()
        pending += 1
        if pending == update_period:
            apply_gradients(model, optimizer, grad_clip)
            pending = 0
        running_loss += loss.item()
        if batch_index % LOG_EVERY == 0 or batch_index == len(loader):
            print(f"batch={batch_index}/{len(loader)} loss={running_loss / batch_index:.6f} "
                  f"lr={optimizer.param_groups[0]['lr']:.6g} "
                  f"speed={batch_index / max(time.time() - start, 1e-9):.2f} batch/s", flush=True)
    if pending:  # the last batches of the epoch still update the model
        apply_gradients(model, optimizer, grad_clip)
    return running_loss / len(loader)


def load_dev_set(dev_dir: Path) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Every dev sample as its own batch of one."""
    # one sample per batch: the model's attention has no padding mask, so padded batches would change predictions
    return [load_batch(dev_dir, [sample_id]) for sample_id, _ in read_index(dev_dir)]


@torch.no_grad()
def dev_metrics(model: RT_est, dev_set: list[tuple[torch.Tensor, torch.Tensor]], device: torch.device) -> dict:
    """Bias (label minus prediction), MAE, MSE and Pearson rho of the model on the dev set."""
    was_training = model.training
    model.eval()
    predictions = np.array([float(model(rates.to(device, torch.float32))[0].cpu().view(-1)[0])
                            for rates, _ in dev_set])
    model.train(was_training)
    targets = np.array([float(labels[0]) for _, labels in dev_set])
    errors = targets - predictions
    return {"dev_bias": float(np.mean(errors)), "dev_mae": float(np.mean(np.abs(errors))),
            "dev_mse": float(np.mean(errors ** 2)),
            "dev_pearson_rho": float(np.corrcoef(targets, predictions)[0, 1]) if len(dev_set) > 1 else float("nan"),
            "dev_n": len(dev_set)}


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True, help="training feature folder")
    parser.add_argument("--dev", type=Path, required=True, help="validation feature folder, scored every epoch")
    parser.add_argument("--out", type=Path, required=True, help="new run folder for weights and logs")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--fc-dim", type=int, default=512)
    parser.add_argument("--decay-epoch", type=int, default=50, help="decay the learning rate after this epoch")
    parser.add_argument("--decay-lr", type=float, default=0.95, help="learning-rate factor per epoch after --decay-epoch")
    parser.add_argument("--update-period", type=int, default=16, help="batches summed per optimizer step")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--save-epochs-from", type=int, default=40,
                        help="also keep each epoch's weights from this epoch on, for averaging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = RT_est(num_channels=N_MELS, fc_dim=args.fc_dim).to(device)
    model.apply(init_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.decay_lr)
    batches = length_batches(args.train, args.batch_size)
    batch_order = torch.Generator().manual_seed(args.seed)
    dev_set = load_dev_set(args.dev)

    args.out.mkdir(parents=True)  # refuses an existing run folder, so no earlier run is overwritten
    epochs_dir = args.out / "epoch_checkpoints"
    epochs_dir.mkdir()
    (args.out / "run.json").write_text(json.dumps(vars(args), indent=2, default=str) + "\n")
    with (args.out / "train_log.csv").open("w", newline="") as handle:
        log = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        log.writeheader()
        for epoch in range(1, args.epochs + 1):
            print(f"epoch {epoch}/{args.epochs}", flush=True)
            loader = epoch_loader(args.train, batches, batch_order, args.num_workers)
            train_loss = train_epoch(model, optimizer, loader, args.update_period, args.grad_clip, device)
            model.save(args.out / "latest_weights.pyt")
            if epoch >= args.save_epochs_from:
                model.save(epochs_dir / f"epoch{epoch}_weights.pyt")
            if epoch > args.decay_epoch:
                scheduler.step()
            row = {"epoch": epoch, "train_loss": train_loss, **dev_metrics(model, dev_set, device)}
            log.writerow(row)
            handle.flush()
            print(row, flush=True)


if __name__ == "__main__":
    main()
