"""Train the evidential error model and log GT-free uncertainty calibration.

Run (synthetic, reproducible, ~minutes on an RTX 4080):

    python -m rl_nninteractive.train_evidential --data synthetic \
        --epochs 30 --train-size 480 --val-size 96 \
        --out-dir <offload>/ohif-ai-edl

The calibration metrics logged each epoch are the honesty check for this
project: the whole point of EDL here is that the uncertainty is a *trustworthy,
ground-truth-free* stand-in for "where is the mask wrong". We validate that with
AUROC(uncertainty -> voxel-error) and ECE of the Dirichlet mean.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise ImportError("training requires PyTorch") from exc

from .evidential import (
    EvidentialErrorNet3D,
    dirichlet_alpha,
    dirichlet_uncertainty,
    evidential_segmentation_loss,
    inverse_frequency_class_weights,
    set_seed,
)
from .evidential_dataset import SyntheticEvidentialDataset


# --------------------------------------------------------------------------- #
# Calibration metrics (numpy, no sklearn dependency)
# --------------------------------------------------------------------------- #
def auroc(scores: np.ndarray, labels: np.ndarray, *, max_points: int = 2_000_000) -> float:
    """AUROC via the Mann-Whitney U statistic. ``labels`` is a boolean array.

    Subsamples to ``max_points`` for tractability on large volumes (seeded).
    """

    scores = np.asarray(scores, dtype=np.float64).ravel()
    labels = np.asarray(labels).ravel().astype(bool)
    n = scores.size
    if n > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        scores, labels = scores[idx], labels[idx]
    pos = labels.sum()
    neg = labels.size - pos
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    # average ranks for ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    csum = np.cumsum(counts)
    avg = csum - (counts - 1) / 2.0
    ranks = avg[inv]
    sum_pos = ranks[labels].sum()
    u = sum_pos - pos * (pos + 1) / 2.0
    return float(u / (pos * neg))


def expected_calibration_error(confidence: np.ndarray, correct: np.ndarray, *, bins: int = 10) -> float:
    """Standard ECE of predicted confidence vs empirical accuracy."""

    confidence = np.asarray(confidence, dtype=np.float64).ravel()
    correct = np.asarray(correct).ravel().astype(np.float64)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    n = confidence.size
    for b in range(bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (confidence > lo) & (confidence <= hi) if b > 0 else (confidence >= lo) & (confidence <= hi)
        if not mask.any():
            continue
        acc = correct[mask].mean()
        conf = confidence[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    val_data: float
    val_kl: float
    auroc_perror: float     # p_error ranks voxel errors
    auroc_vacuity: float    # epistemic vacuity ranks voxel errors
    ece: float
    seconds: float


# --------------------------------------------------------------------------- #
# Eval pass
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(model, loader, device, *, epoch: int) -> dict[str, float]:
    model.eval()
    losses, datas, kls = [], [], []
    perror_all, vac_all, iserr_all = [], [], []
    conf_all, corr_all = [], []
    for batch in loader:
        x = batch["input"].to(device)
        labels = batch["labels"].to(device)
        evidence = model(x)
        out = evidential_segmentation_loss(evidence, labels, epoch=epoch)
        losses.append(float(out["loss"]))
        datas.append(float(out["data"]))
        kls.append(float(out["kl"]))
        alpha = dirichlet_alpha(evidence)
        u = dirichlet_uncertainty(alpha)
        prob = u["prob"]
        p_error = u["p_error"].detach().cpu().numpy()
        vac = u["vacuity"].detach().cpu().numpy()
        is_err = (labels != 0).detach().cpu().numpy()
        conf = prob.max(dim=1).values.detach().cpu().numpy()
        pred = prob.argmax(dim=1).detach().cpu().numpy()
        corr = (pred == labels.detach().cpu().numpy())
        # subsample per batch to keep memory bounded
        perror_all.append(p_error.ravel()[::4]); vac_all.append(vac.ravel()[::4])
        iserr_all.append(is_err.ravel()[::4])
        conf_all.append(conf.ravel()[::16]); corr_all.append(corr.ravel()[::16])
    perror = np.concatenate(perror_all); vac = np.concatenate(vac_all); iserr = np.concatenate(iserr_all)
    conf = np.concatenate(conf_all); corr = np.concatenate(corr_all)
    return {
        "val_loss": float(np.mean(losses)),
        "val_data": float(np.mean(datas)),
        "val_kl": float(np.mean(kls)),
        "auroc_perror": auroc(perror, iserr),
        "auroc_vacuity": auroc(vac, iserr),
        "ece": expected_calibration_error(conf, corr),
    }


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #
def train(args: argparse.Namespace) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    if args.data == "synthetic":
        train_ds = SyntheticEvidentialDataset(args.train_size, base_seed=args.seed)
        val_ds = SyntheticEvidentialDataset(args.val_size, base_seed=args.seed + 10_000_000)
    elif args.data == "msd":
        from .evidential_dataset import (
            InMemoryEvidentialDataset,
            find_msd_cases,
            materialize_msd_samples,
        )

        pairs = find_msd_cases(args.msd_root)
        n_val = max(1, len(pairs) // 5)
        val_pairs, train_pairs = pairs[:n_val], pairs[n_val:]
        print(f"[edl] materializing MSD patches: {len(train_pairs)} train / {len(val_pairs)} val cases "
              f"x {args.samples_per_case} samples ...", flush=True)
        train_ds = InMemoryEvidentialDataset(
            materialize_msd_samples(train_pairs, base_seed=args.seed, samples_per_case=args.samples_per_case)
        )
        val_ds = InMemoryEvidentialDataset(
            materialize_msd_samples(val_pairs, base_seed=args.seed + 10_000_000, samples_per_case=args.samples_per_case)
        )
    else:
        raise ValueError(f"unknown data source: {args.data}")

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)

    model = EvidentialErrorNet3D(in_channels=2, base_channels=args.base_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    out_dir = Path(args.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    runs_dir = out_dir / "runs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = args.run_id or f"edl_{args.data}_{int(time.time())}"
    log_path = runs_dir / f"{run_id}.jsonl"
    best_ckpt = ckpt_dir / f"{run_id}_best.pt"
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[edl] device={device} params={n_params/1e6:.2f}M amp={use_amp} "
          f"train={len(train_ds)} val={len(val_ds)} run={run_id}", flush=True)

    best_score = -float("inf")
    history: list[dict[str, Any]] = []
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        train_losses = []
        for batch in train_loader:
            x = batch["input"].to(device)
            labels = batch["labels"].to(device)
            cw = inverse_frequency_class_weights(labels).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                evidence = model(x)
                out = evidential_segmentation_loss(
                    evidence, labels, epoch=epoch, anneal_epochs=args.anneal_epochs, class_weights=cw
                )
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_losses.append(float(loss.detach()))
        val = evaluate(model, val_loader, device, epoch=epoch)
        secs = time.time() - t0
        m = EpochMetrics(
            epoch=epoch,
            train_loss=float(np.mean(train_losses)),
            val_loss=val["val_loss"],
            val_data=val["val_data"],
            val_kl=val["val_kl"],
            auroc_perror=val["auroc_perror"],
            auroc_vacuity=val["auroc_vacuity"],
            ece=val["ece"],
            seconds=secs,
        )
        history.append(asdict(m))
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(asdict(m)) + "\n")
        print(f"[edl] ep{epoch:02d} train={m.train_loss:.4f} val={m.val_loss:.4f} "
              f"AUROC(p_err)={m.auroc_perror:.3f} AUROC(vac)={m.auroc_vacuity:.3f} "
              f"ECE={m.ece:.3f} ({secs:.1f}s)", flush=True)

        score = m.auroc_perror if np.isfinite(m.auroc_perror) else -m.val_loss
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": vars(args),
                    "epoch": epoch,
                    "metrics": asdict(m),
                    "in_channels": 2,
                    "base_channels": args.base_channels,
                },
                best_ckpt,
            )

    summary = {
        "run_id": run_id,
        "best_auroc_perror": best_score,
        "best_ckpt": str(best_ckpt),
        "log": str(log_path),
        "final": history[-1] if history else None,
        "n_params": n_params,
    }
    (runs_dir / f"{run_id}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[edl] DONE best_AUROC(p_err)={best_score:.3f} ckpt={best_ckpt}", flush=True)
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train evidential error model")
    p.add_argument("--data", choices=["synthetic", "msd"], default="synthetic")
    p.add_argument("--msd-root", default="")
    p.add_argument("--samples-per-case", type=int, default=6)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--train-size", type=int, default=480)
    p.add_argument("--val-size", type=int, default=96)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--anneal-epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", default="")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
