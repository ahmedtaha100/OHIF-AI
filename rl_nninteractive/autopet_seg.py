"""Promptable evidential PET/CT lesion segmenter (the AutoPET-proven recipe).

Implements the approach that actually wins interactive whole-body PET/CT lesion
segmentation (autoPET IV winner; arXiv 2508.21680): a segmenter that takes
CT + PET plus foreground/background prompt channels and is trained with
stochastic online clicks, so ONE model serves the whole 0..k-click curve.
Automated -> +5 clicks buys ~+8 Dice, almost entirely by recovering missed
lesions (false negatives) -- the clinician-time win.

Our additions vs the plain recipe:
  * the segmenter is *evidential* (per-voxel Dirichlet over {background, lesion})
    so it emits calibrated uncertainty, not just a mask;
  * that uncertainty drives a **ground-truth-free** next-click targeting policy
    (place a foreground prompt at the highest SUV x uncertainty voxel not yet
    segmented = the most likely missed lesion). Compared here against the GT
    oracle (click the true largest missed lesion) and no interaction.

Prompts are encoded as a (proven-better-than-Gaussian) Euclidean-distance field.
Sized for a 16 GB RTX 4080: compact U-Net, 96^3 patches, whole-volume inference
on a fixed resampled grid.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy import ndimage

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    raise ImportError("autopet_seg requires PyTorch") from exc

from .evidential import (
    EvidentialErrorNet3D,
    dirichlet_alpha,
    evidential_segmentation_loss,
    inverse_frequency_class_weights,
    set_seed,
)
from .autopet_pipeline import find_autopet_cases, load_autopet_volume, lesion_components, _representative_coord

WORK_GRID = (256, 128, 128)   # (Z,Y,X) fixed resampled grid
PATCH = (96, 96, 96)
LESION = 1
BG = 0


# --------------------------------------------------------------------------- #
# Prompt encoding (EDT field, proven > Gaussian)
# --------------------------------------------------------------------------- #
def edt_field(shape: tuple[int, int, int], points: list[tuple[int, int, int]], *, falloff: float = 20.0) -> np.ndarray:
    """Normalized proximity field: 1 at a click, linearly fading over `falloff` voxels."""

    if not points:
        return np.zeros(shape, dtype=np.float32)
    seed = np.ones(shape, dtype=bool)
    for p in points:
        if all(0 <= p[a] < shape[a] for a in range(3)):
            seed[p] = False
    dist = ndimage.distance_transform_edt(seed)
    return np.clip(1.0 - dist / falloff, 0.0, 1.0).astype(np.float32)


def _sample_click(mask: np.ndarray, rng) -> tuple[int, int, int] | None:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return None
    c = coords[int(rng.integers(0, len(coords)))]
    return (int(c[0]), int(c[1]), int(c[2]))


def simulate_training_clicks(gt: np.ndarray, rng, *, max_clicks: int = 10):
    """Stochastic clicks favoring few, FG-weighted (FN recovery is the win)."""

    # bias toward click-conditioned samples so the model LEARNS to use prompts:
    # 15% automated (k=0), else 1..max_clicks (favoring few).
    if rng.random() < 0.15:
        k = 0
    else:
        k = int(1 + rng.integers(0, max_clicks))
        k = min(k, int(1 + abs(rng.normal(1.5, 1.5))))  # skew toward few clicks
    fg_pts, bg_pts = [], []
    comps = lesion_components(gt, min_size=3)
    bg = ~gt
    for _ in range(k):
        if comps and rng.random() < 0.75:            # 75% foreground clicks
            comp = comps[int(rng.integers(0, len(comps)))]
            p = _sample_click(comp, rng)
            if p:
                fg_pts.append(p)
        else:
            p = _sample_click(bg, rng)
            if p:
                bg_pts.append(p)
    return fg_pts, bg_pts


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
def _roi(shape, center, patch):
    out = []
    for a in range(3):
        s = min(patch[a], shape[a])
        lo = int(np.clip(center[a] - s // 2, 0, shape[a] - s))
        out.append((lo, lo + s))
    return out


def _pad_to(a, target, pad_value):
    pads = [(0, max(0, target[i] - a.shape[i])) for i in range(3)]
    if all(hi == 0 for _, hi in pads):
        return a
    return np.pad(a, pads, mode="constant", constant_values=pad_value)


class AutoPetPatchDataset(Dataset):
    """Lesion-centred + random patches with online stochastic click channels."""

    def __init__(self, volumes: list, *, base_seed: int = 0, patches_per_case: int = 8):
        self.vols = volumes
        self.base_seed = base_seed
        self.ppc = patches_per_case

    def __len__(self):
        return len(self.vols) * self.ppc

    def __getitem__(self, index):
        v = self.vols[index % len(self.vols)]
        rng = np.random.default_rng(self.base_seed + index)
        comps = lesion_components(v.gt, min_size=3)
        if comps and rng.random() < 0.8:
            center = _representative_coord(comps[int(rng.integers(0, len(comps)))])
        else:
            center = tuple(int(rng.integers(0, s)) for s in v.gt.shape)
        b = _roi(v.gt.shape, center, PATCH)
        sl = tuple(slice(lo, hi) for lo, hi in b)
        ct = _pad_to(v.ct[sl], PATCH, -1.0)
        pet = _pad_to(v.pet[sl], PATCH, -1.0)
        gt = _pad_to(v.gt[sl].astype(np.uint8), PATCH, 0).astype(bool)
        fg_pts, bg_pts = simulate_training_clicks(gt, rng)
        fg = edt_field(PATCH, fg_pts)
        bg = edt_field(PATCH, bg_pts)
        x = np.stack([ct, pet, fg, bg], axis=0).astype(np.float32)
        return {"input": torch.from_numpy(x), "target": torch.from_numpy(gt.astype(np.int64))}


# --------------------------------------------------------------------------- #
# Loss: evidential + soft Dice (no smoothing, per winner recipe)
# --------------------------------------------------------------------------- #
def soft_dice_loss(prob_lesion: "torch.Tensor", target: "torch.Tensor") -> "torch.Tensor":
    p = prob_lesion.reshape(prob_lesion.shape[0], -1)
    y = (target == LESION).float().reshape(target.shape[0], -1)
    inter = (p * y).sum(1)
    denom = p.sum(1) + y.sum(1)
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return (1.0 - dice).mean()


# --------------------------------------------------------------------------- #
# Inference (whole volume, click-conditioned)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def infer(model, ct, pet, fg_pts, bg_pts, device):
    fg = edt_field(ct.shape, fg_pts)
    bg = edt_field(ct.shape, bg_pts)
    x = np.stack([ct, pet, fg, bg], axis=0)[None]
    t = torch.from_numpy(x.astype(np.float32)).to(device)
    ev = model(t)
    alpha = dirichlet_alpha(ev)
    S = alpha.sum(1, keepdim=True)
    prob = (alpha / S)[0].cpu().numpy()          # (2,Z,Y,X)
    vac = (2.0 / S)[0, 0].cpu().numpy()           # K/S vacuity
    p_les = prob[LESION]
    return p_les, vac


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred.astype(bool); g = gt.astype(bool)
    d = 2 * np.logical_and(p, g).sum() / (p.sum() + g.sum() + 1e-8)
    return float(d)


@torch.no_grad()
def add_corrective_clicks(x: "torch.Tensor", y: "torch.Tensor", model, *, rng, falloff: float = 20.0,
                          n_rounds: int = 1) -> "torch.Tensor":
    """Plant foreground clicks at the model's OWN false-negatives (corrective training).

    Placing prompts where the current prediction misses a lesion is what teaches
    the model to *respond* to clicks (DeepEdit/nnInteractive-style) -- clicks at
    already-segmented lesions are redundant and get ignored.
    """

    x = x.clone()
    gt = y.detach().cpu().numpy()
    for _ in range(n_rounds):
        ev = model(x)
        alpha = dirichlet_alpha(ev)
        pred = (alpha[:, LESION] / alpha.sum(1)).detach().cpu().numpy() >= 0.5
        fg = x[:, 2].detach().cpu().numpy()
        changed = False
        for b in range(fg.shape[0]):
            fn = (gt[b] == LESION) & (~pred[b])
            comps = lesion_components(fn, min_size=5)
            if not comps:
                continue
            comp = comps[int(rng.integers(0, len(comps)))] if rng.random() < 0.3 else max(comps, key=lambda c: int(c.sum()))
            ball = edt_field(fg[b].shape, [_representative_coord(comp)], falloff=falloff)
            fg[b] = np.maximum(fg[b], ball)
            changed = True
        if not changed:
            break
        x[:, 2] = torch.from_numpy(fg).to(x.device)
    return x


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args) -> dict[str, Any]:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = find_autopet_cases(args.data_root, tracer=args.tracer)
    print(f"[autopet-seg] {len(cases)} {args.tracer} cases; loading volumes ...", flush=True)
    vols = []
    for c in cases[: args.n_cases]:
        try:
            vols.append(load_autopet_volume(c, target=WORK_GRID))
        except Exception as e:
            print("  skip", c.case_id, e, flush=True)
    n_val = max(2, len(vols) // 5)
    val_vols, train_vols = vols[:n_val], vols[n_val:]
    print(f"[autopet-seg] train {len(train_vols)} / val {len(val_vols)} volumes", flush=True)

    ds = AutoPetPatchDataset(train_vols, base_seed=args.seed, patches_per_case=args.patches_per_case)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0, drop_last=True)
    model = EvidentialErrorNet3D(in_channels=4, base_channels=args.base_channels, num_classes=2).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    ckpt = Path(args.out_dir) / "checkpoints" / f"{args.run_id}_best.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[autopet-seg] params={n_params/1e6:.2f}M patch={PATCH} grid={WORK_GRID}", flush=True)

    for epoch in range(args.epochs):
        model.train(); t0 = time.time(); losses = []
        for batch in dl:
            x = batch["input"].to(device); y = batch["target"].to(device)
            if epoch >= args.corrective_start:                      # corrective clicks at the model's misses
                model.eval()
                x = add_corrective_clicks(x, y, model, rng=np.random.default_rng(args.seed * 97 + epoch))
                model.train()
            opt.zero_grad(set_to_none=True)
            cw = inverse_frequency_class_weights(y, num_classes=2).to(device)  # counter ~0.1% lesion voxels
            with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
                ev = model(x)
                out = evidential_segmentation_loss(ev, y, epoch=epoch, anneal_epochs=args.anneal_epochs, class_weights=cw)
                alpha = dirichlet_alpha(ev); p_les = (alpha[:, LESION] / alpha.sum(1))
                loss = out["loss"] + args.dice_weight * soft_dice_loss(p_les, y)
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            losses.append(float(loss.detach()))
        # validation (interactive is expensive) every val_every epochs + last.
        if (epoch % args.val_every == 0) or (epoch == args.epochs - 1):
            model.eval()
            res = run_interactive_eval(model, val_vols[: args.val_cases], device, ks=(0, 5), mode="oracle")
            auto, inter = res[0], res[5]
            print(f"[autopet-seg] ep{epoch:02d} loss={np.mean(losses):.4f} val_auto={auto:.4f} "
                  f"val_inter@5={inter:.4f} ({time.time()-t0:.1f}s)", flush=True)
            if inter > best:                   # select by INTERACTIVE performance (the deliverable)
                best = inter
                torch.save({"model_state": model.state_dict(), "in_channels": 4, "num_classes": 2,
                            "base_channels": args.base_channels, "epoch": epoch,
                            "val_auto_dice": auto, "val_inter_dice": inter}, ckpt)
        else:
            print(f"[autopet-seg] ep{epoch:02d} loss={np.mean(losses):.4f} ({time.time()-t0:.1f}s)", flush=True)
    print(f"[autopet-seg] DONE best_auto_Dice={best:.4f} ckpt={ckpt}", flush=True)
    return {"best_auto_dice": best, "ckpt": str(ckpt), "n_val": len(val_vols)}


def load_seg_model(ckpt_path, device):
    ck = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    m = EvidentialErrorNet3D(in_channels=4, base_channels=int(ck.get("base_channels", 24)), num_classes=2).to(device)
    m.load_state_dict(ck["model_state"]); m.eval()
    return m


def _next_click_oracle(gt, pred):
    missed = gt & ~pred
    comps = lesion_components(missed, min_size=3)
    if not comps:
        return None
    return _representative_coord(max(comps, key=lambda c: int(c.sum())))


def _next_click_edl(pet, vac, pred, *, pet_pct=90.0):
    """GT-free: brightest * most-uncertain voxel not yet segmented = likely missed lesion."""
    cand = (~pred) & (pet >= np.percentile(pet, pet_pct))
    if not cand.any():
        return None
    score = (pet + 1.0) * 0.5 * vac      # both in ~[0,1]
    score[~cand] = -1.0
    idx = int(np.argmax(score))
    return tuple(int(x) for x in np.unravel_index(idx, score.shape))


@torch.no_grad()
def run_interactive_eval(model, vols, device, *, ks=(0, 1, 3, 5, 8), mode="edl"):
    """Return per-k mean Dice over held-out volumes; clicks placed by `mode`.

    mode='oracle' clicks the true largest missed lesion (upper bound);
    mode='edl' clicks the highest SUVxuncertainty unsegmented voxel (GT-free).
    """
    maxk = max(ks)
    curves = {k: [] for k in ks}
    for v in vols:
        fg_pts = []
        p_les, vac = infer(model, v.ct, v.pet, fg_pts, [], device)
        pred = p_les >= 0.5
        if 0 in curves:
            curves[0].append(dice(pred, v.gt))
        for step in range(1, maxk + 1):
            if mode == "oracle":
                c = _next_click_oracle(v.gt, pred)
            else:
                c = _next_click_edl(v.pet, vac, pred)
            if c is not None:
                fg_pts.append(c)
            p_les, vac = infer(model, v.ct, v.pet, fg_pts, [], device)
            pred = p_les >= 0.5
            if step in curves:
                curves[step].append(dice(pred, v.gt))
    return {k: float(np.mean(curves[k])) for k in ks}


def build_parser():
    p = argparse.ArgumentParser(description="Train promptable evidential PET/CT lesion segmenter")
    p.add_argument("--data-root", required=True)
    p.add_argument("--tracer", default="FDG")
    p.add_argument("--n-cases", type=int, default=40)
    p.add_argument("--patches-per-case", type=int, default=8)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dice-weight", type=float, default=5.0)
    p.add_argument("--anneal-epochs", type=int, default=10)
    p.add_argument("--corrective-start", type=int, default=3, help="epoch to begin corrective-click training")
    p.add_argument("--val-every", type=int, default=3)
    p.add_argument("--val-cases", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", default="autopet_seg_fdg_v1")
    return p


if __name__ == "__main__":
    train(build_parser().parse_args())
