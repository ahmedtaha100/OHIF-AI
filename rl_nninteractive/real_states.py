"""Train the evidential model on REAL nnInteractive rollout states (sim-to-real).

The v1 evidential model was trained on synthetic *perturbed* masks (eroded/
dilated ground truth). On real nnInteractive output — which is already ~0.89
Dice from one click — that model is out-of-distribution: it over-predicts error
and keeps clicking, hurting Dice (see runs/rollout_edl_v1.json).

This module closes that gap. It drives real nnInteractive on the *training*
cases to capture the masks it actually produces (1 click = good; +1 random click
= a degraded state with a genuine nnInteractive-shaped error), labels each state
against ground truth, and retrains the evidential model on that real
distribution. The retrained model learns that nnInteractive masks are mostly
correct — so it stops early — while still localizing the residual real errors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
    from torch.utils.data import DataLoader
except Exception as exc:  # pragma: no cover
    raise ImportError("real_states requires PyTorch") from exc

from .evidential import (
    EvidentialErrorNet3D,
    evidential_segmentation_loss,
    inverse_frequency_class_weights,
    set_seed,
)
from .evidential_dataset import InMemoryEvidentialDataset, _pad_to, build_sample, find_msd_cases
from .metrics import dice_score
from .real_rollout import DEFAULT_MODEL_ROOT, _roi_bounds, _representative_coord, load_case_zyx, make_session, _add_point
from .robot_user import largest_component_robot_action
from .train_evidential import evaluate
from .env import POINT_POSITIVE, STOP
from .provenance import CacheIdentity, make_cache_envelope, sha256_json, unwrap_cache_envelope


def _crop_state(win: np.ndarray, gt: np.ndarray, mask: np.ndarray, *, size: int):
    """Center a fixed `size^3` crop on the tumor (or volume center) and pad if needed."""

    center = _representative_coord(gt) if gt.any() else tuple(d // 2 for d in gt.shape)
    b = _roi_bounds(gt.shape, center, (size, size, size))
    sl = tuple(slice(lo, hi) for lo, hi in b)
    ts = (size, size, size)
    return (
        _pad_to(win[sl], ts, pad_value=-1.0),
        _pad_to(gt[sl], ts, pad_value=False),
        _pad_to(mask[sl].astype(bool), ts, pad_value=False),
    )


def generate_real_state_samples(
    session,
    pairs: Sequence[tuple[Path, Path]],
    *,
    patch: int = 128,
    train_patch: int = 64,
    seeds_per_case: int = 3,
    device: str = "cuda",
    seed0: int = 0,
    tumor_label: int = 1,
) -> list[Any]:
    """Capture (image, real nnInteractive mask, gt) states as EvidentialSamples.

    nnInteractive runs on a `patch^3` ROI for context; each stored training
    sample is a `train_patch^3` crop centred on the tumor (keeps memory small and
    matches the evidential model's training scale).
    """

    samples: list[Any] = []
    for ci, (img_path, lab_path) in enumerate(pairs):
        raw, win, gt = load_case_zyx(img_path, lab_path, tumor_label=tumor_label)
        if not gt.any():
            continue
        tumor = np.argwhere(gt)
        rng = np.random.default_rng(seed0 + ci)
        for si in range(seeds_per_case):
            seed_full = tuple(int(v) for v in tumor[rng.integers(0, len(tumor))])
            b = _roi_bounds(gt.shape, seed_full, (patch, patch, patch))
            sl = tuple(slice(lo, hi) for lo, hi in b)
            roi_raw, roi_win, roi_gt = raw[sl], win[sl], gt[sl]
            if not roi_gt.any():
                continue
            seed = tuple(seed_full[a] - b[a][0] for a in range(3))

            session.reset_interactions()
            session.set_image(roi_raw[None])
            session.set_target_buffer(torch.zeros(roi_raw.shape, dtype=torch.uint8))
            mask = _add_point(session, seed, positive=True)          # 1-click (good) state
            w, g, m = _crop_state(roi_win, roi_gt, mask, size=train_patch)
            samples.append(build_sample(w, g, m, case_id=f"real_{ci}_{si}_seed"))

            # a second, deliberately noisy click -> a degraded state with real error to localize
            noisy = tuple(int(s) for s in rng.integers(0, np.array(roi_gt.shape)))
            positive = not bool(mask[noisy])
            mask2 = _add_point(session, noisy, positive=positive)
            w2, g2, m2 = _crop_state(roi_win, roi_gt, mask2, size=train_patch)
            samples.append(build_sample(w2, g2, m2, case_id=f"real_{ci}_{si}_noisy"))
    return samples


def _train_loop(train_ds, val_ds, *, args, device) -> dict[str, Any]:
    set_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0)
    model = EvidentialErrorNet3D(in_channels=2, base_channels=args.base_channels).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    ckpt = Path(args.out_dir) / "checkpoints" / f"{args.run_id}_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        losses = []
        for batch in train_loader:
            x = batch["input"].to(device); labels = batch["labels"].to(device)
            cw = inverse_frequency_class_weights(labels).to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device.type, enabled=use_amp):
                out = evidential_segmentation_loss(model(x), labels, epoch=epoch, anneal_epochs=args.anneal_epochs, class_weights=cw)
            scaler.scale(out["loss"]).backward(); scaler.step(opt); scaler.update()
            losses.append(float(out["loss"].detach()))
        val = evaluate(model, val_loader, device, epoch=epoch)
        score = val["auroc_perror"] if np.isfinite(val["auroc_perror"]) else -val["val_loss"]
        print(f"[real-edl] ep{epoch:02d} train={np.mean(losses):.4f} val={val['val_loss']:.4f} "
              f"AUROC(p_err)={val['auroc_perror']:.3f} ECE={val['ece']:.3f} ({time.time()-t0:.1f}s)", flush=True)
        if score > best:
            best = score
            torch.save({"model_state": model.state_dict(), "in_channels": 2,
                        "base_channels": args.base_channels, "epoch": epoch, "metrics": val}, ckpt)
    print(f"[real-edl] DONE best_AUROC(p_err)={best:.3f} ckpt={ckpt}", flush=True)
    return {"best_auroc": best, "ckpt": str(ckpt)}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description="Retrain evidential model on real nnInteractive states")
    p.add_argument("--msd-root", required=True)
    p.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    p.add_argument("--n-train-cases", type=int, default=40)
    p.add_argument("--seeds-per-case", type=int, default=3)
    p.add_argument("--tumor-label", type=int, default=1, help="MSD label id for tumor (Pancreas cancer=2)")
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--train-patch", type=int, default=64)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=16)
    p.add_argument("--anneal-epochs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", default="edl_real_lung_v2")
    p.add_argument("--states-out", default="")
    p.add_argument("--states-cache", default="", help="path to cache/reuse captured states (skip nnInteractive if present)")
    p.add_argument("--checkpoint-sha256", default="", help="required when --states-cache is used")
    p.add_argument("--dataset-sha256", default="", help="required when --states-cache is used")
    args = p.parse_args(argv)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pairs = find_msd_cases(args.msd_root)
    n_val = max(1, len(pairs) // 5)
    train_pairs = pairs[n_val:][: args.n_train_cases]   # held-out (first n_val) reserved for rollout eval

    cache = Path(args.states_cache) if args.states_cache else None
    cache_identity = None
    if cache:
        cache_identity = CacheIdentity(
            namespace="real_nninteractive_state_samples",
            case_ids=tuple(str(image_path.resolve()) for image_path, _ in train_pairs),
            target_label=str(args.tumor_label),
            checkpoint_sha256=args.checkpoint_sha256,
            dataset_sha256=args.dataset_sha256,
            config_sha256=sha256_json(
                {
                    "patch": args.patch,
                    "train_patch": args.train_patch,
                    "seeds_per_case": args.seeds_per_case,
                    "seed": args.seed,
                    "tumor_label": args.tumor_label,
                }
            ),
        )
    if cache and cache.exists():
        print(f"[real-edl] loading cached states from {cache}", flush=True)
        envelope = torch.load(str(cache), weights_only=False)
        samples = unwrap_cache_envelope(envelope, cache_identity)
    else:
        print(f"[real-edl] loading nnInteractive; capturing states from {len(train_pairs)} training cases ...", flush=True)
        session = make_session(device=str(device), model_root=args.model_root)
        t0 = time.time()
        samples = generate_real_state_samples(
            session, train_pairs, patch=args.patch, train_patch=args.train_patch,
            seeds_per_case=args.seeds_per_case, device=str(device), seed0=args.seed, tumor_label=args.tumor_label
        )
        print(f"[real-edl] captured {len(samples)} real states in {time.time()-t0:.0f}s", flush=True)
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(make_cache_envelope(cache_identity, samples), str(cache))
            print(f"[real-edl] cached states -> {cache}", flush=True)

    # report the real-state mask quality distribution (context for the writeup)
    dices = [float(dice_score(s.current_mask, s.gt)) for s in samples]
    print(f"[real-edl] real-state Dice: min={min(dices):.3f} mean={np.mean(dices):.3f} max={max(dices):.3f}", flush=True)

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(samples))
    n_v = max(1, len(samples) // 5)
    val_ds = InMemoryEvidentialDataset([samples[i] for i in idx[:n_v]])
    train_ds = InMemoryEvidentialDataset([samples[i] for i in idx[n_v:]])
    print(f"[real-edl] train={len(train_ds)} val={len(val_ds)}", flush=True)

    result = _train_loop(train_ds, val_ds, args=args, device=device)
    if args.states_out:
        Path(args.states_out).write_text(json.dumps({"n_samples": len(samples), "dice_stats": {
            "min": min(dices), "mean": float(np.mean(dices)), "max": max(dices)}}, indent=2), encoding="utf-8")
    return result


if __name__ == "__main__":
    main()
