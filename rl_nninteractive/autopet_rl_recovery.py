"""RL + evidential-DL multifocal lesion recovery for whole-body PET/CT.

Design (the tractable, candidate-recovery path after voxel-segmentation from
scratch proved too hard on limited data):

  1. Candidate components = PET-hot connected components (GT-free).
  2. An **evidential classifier** scores each candidate lesion-vs-physiological
     from component features (SUV, CT, size, organ overlap, location) -> P(lesion)
     + uncertainty. This is the evidential-DL contribution and it directly solves
     the false-positive problem that sank the intensity heuristics.
  3. An **RL policy** sequences the clicks: at each step it accepts or skips the
     next candidate (nnInteractive segments accepted candidates locally -> union),
     using evidential scores as state. Reward = whole-body dDice - click cost.
  4. Clicking a real lesion raises Dice (monotone-safe); the evidential score lets
     the policy skip physiological uptake. Compared vs 1-click, click-all,
     evidential-greedy, and the GT oracle.

The classifier is small (component feature vector -> Dirichlet), so it trains
reliably on limited data, unlike voxel segmentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage, stats

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise ImportError("autopet_rl_recovery requires PyTorch") from exc

from .evidential import (
    dirichlet_alpha,
    evidential_segmentation_loss,
    inverse_frequency_class_weights,
    set_seed,
)
from .autopet_pipeline import load_autopet_volume, _representative_coord
from .medical_geometry import load_nifti_on_reference_grid
from .provenance import (
    CacheIdentity,
    make_cache_envelope,
    sha256_file,
    sha256_json,
    unwrap_cache_envelope,
)

STR = np.ones((3, 3, 3), dtype=bool)
FEAT_NAMES = ("log_size", "mean_suv", "max_suv", "ct_mean", "ct_std",
              "z_center", "y_center", "x_center", "organ_overlap", "compactness", "suv_p90")
FEAT_DIM = len(FEAT_NAMES)


def load_totseg(case, target):
    aligned = load_nifti_on_reference_grid(
        case.ct.parent / "totseg_24.nii.gz",
        reference_path=case.pet,
        target_shape_zyx=target,
        is_label=True,
    )
    return aligned.data_zyx


@dataclass
class Candidate:
    coord: tuple[int, int, int]
    mask: np.ndarray
    features: np.ndarray
    label: int          # 1 = lesion (overlaps TTB), 0 = physiological/other


def extract_candidates(
    v,
    totseg,
    *,
    pet_pct: float = 95.0,
    min_size: int = 4,
    max_candidates: int = 60,
    label_candidates: bool = True,
) -> list[Candidate]:
    thr = np.percentile(v.pet, pet_pct)
    hot = v.pet >= thr
    labels, n = ndimage.label(hot, STR)
    if n == 0:
        return []
    sizes = np.bincount(labels.ravel()); sizes[0] = 0
    order = np.argsort(sizes)[::-1]
    shape = np.array(v.pet.shape, dtype=np.float32)
    out: list[Candidate] = []
    for lab in order:
        if lab == 0 or sizes[lab] < min_size or len(out) >= max_candidates:
            continue
        comp = labels == lab
        coords = np.argwhere(comp)
        size = len(coords)
        suv = v.pet[comp]
        ct = v.ct[comp]
        bbox = coords.max(0) - coords.min(0) + 1
        compact = size / float(np.prod(bbox) + 1e-6)
        center = coords.mean(0) / shape
        feats = np.array([
            np.log1p(size), float(suv.mean()), float(suv.max()),
            float(ct.mean()), float(ct.std()),
            float(center[0]), float(center[1]), float(center[2]),
            float((totseg[comp] > 0).mean()), float(compact),
            float(np.percentile(suv, 90)),
        ], dtype=np.float32)
        overlap = float((comp & v.gt).sum()) / size if label_candidates else 0.0
        out.append(Candidate(coord=_representative_coord(comp), mask=comp,
                             features=feats, label=int(overlap > 0.2) if label_candidates else -1))
    return out


class EvidentialCandidateClassifier(nn.Module):
    """Component feature vector -> Dirichlet over {not-lesion, lesion}."""

    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2),
        )

    def forward(self, x):
        return nn.functional.softplus(self.net(x))     # evidence >= 0

    def predict(self, feats: np.ndarray):
        self.eval()
        with torch.no_grad():
            ev = self(torch.from_numpy(np.atleast_2d(feats).astype(np.float32)))
            alpha = dirichlet_alpha(ev)
            S = alpha.sum(1)
            p_les = (alpha[:, 1] / S).numpy()
            vac = (2.0 / S).numpy()
        return p_les, vac


def _auroc(scores, labels):
    scores = np.asarray(scores); labels = np.asarray(labels).astype(bool)
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def train_classifier(train_feats, train_labels, *, epochs=200, lr=1e-3, seed=0):
    set_seed(seed)
    x = torch.from_numpy(train_feats.astype(np.float32))
    y = torch.from_numpy(train_labels.astype(np.int64))
    # normalize features
    mean = x.mean(0, keepdim=True); std = x.std(0, keepdim=True).clamp_min(1e-6)
    xn = (x - mean) / std
    model = EvidentialCandidateClassifier()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    cw = inverse_frequency_class_weights(y, num_classes=2)
    for ep in range(epochs):
        model.train(); opt.zero_grad()
        ev = model(xn)
        # evidential loss expects (B,K,...spatial); treat each component as a 1-voxel "volume"
        out = evidential_segmentation_loss(ev[:, :, None, None, None], y[:, None, None, None],
                                           epoch=ep, anneal_epochs=50, class_weights=cw)
        out["loss"].backward(); opt.step()
    model._mean = mean; model._std = std
    return model


def classify(model, feats: np.ndarray):
    x = (torch.from_numpy(np.atleast_2d(feats).astype(np.float32)) - model._mean) / model._std
    return model.predict(x.numpy())


# --------------------------------------------------------------------------- #
# RL policy over candidates (uses the evidential score as state; nnInteractive
# segmentations precomputed once so REINFORCE is fast). Reward = whole-body dDice.
# --------------------------------------------------------------------------- #
@dataclass
class RecoveryEpisode:
    seed_union: np.ndarray
    gt: np.ndarray
    feats: np.ndarray          # (n, FEAT_DIM)
    P: np.ndarray              # (n,) classifier P(lesion)
    vac: np.ndarray            # (n,) classifier vacuity
    masks: list                # (n,) precomputed local seg masks, sorted by P desc


STATE_DIM = FEAT_DIM + 4       # per-candidate feats + [P, vac, frac_clicked, union_frac]


class RecoveryPolicy(nn.Module):
    def __init__(self, in_dim: int = STATE_DIM, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        nn.init.constant_(self.net[-1].bias, 0.7)   # bias toward clicking so REINFORCE explores +reward

    def forward(self, s):       # -> click logit
        return self.net(s).squeeze(-1)


def _dice(u, g):
    return 2 * np.logical_and(u, g).sum() / (u.sum() + g.sum() + 1e-8)


def _manifest_split_cases(cases, manifest_path, *, train_split, eval_split):
    import json

    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if payload.get("version") != 2:
        raise ValueError("--manifest requires StudyManifest version 2")
    records = [case for dataset in payload.get("datasets", []) for case in dataset.get("cases", [])]
    by_id = {case["case_id"]: case for case in records}
    if len(by_id) != len(records):
        raise ValueError("StudyManifest contains duplicate case_id values")
    discovered = {case.case_id: case for case in cases}
    selected_ids = {
        split: [case_id for case_id, record in by_id.items() if record.get("split") == split]
        for split in (train_split, eval_split)
    }
    missing = sorted(
        case_id for case_ids in selected_ids.values() for case_id in case_ids if case_id not in discovered
    )
    if missing:
        raise ValueError(f"StudyManifest cases are missing from --data-root: {missing}")
    train_cases = [discovered[case_id] for case_id in selected_ids[train_split]]
    eval_cases = [discovered[case_id] for case_id in selected_ids[eval_split]]
    if not train_cases or not eval_cases:
        raise ValueError("manifest-selected train and evaluation splits must both be non-empty")
    patient_by_case = {case_id: record["patient_id"] for case_id, record in by_id.items()}
    train_patients = {patient_by_case[case.case_id] for case in train_cases}
    eval_patients = {patient_by_case[case.case_id] for case in eval_cases}
    leakage = sorted(train_patients & eval_patients)
    if leakage:
        raise ValueError(f"patient leakage across train/evaluation splits: {leakage}")
    return train_cases, eval_cases, patient_by_case


def _paired_patient_statistics(case_ids, patient_by_case, reference, candidate, *, seed=20260715):
    reference = np.asarray(reference, dtype=np.float64)
    candidate = np.asarray(candidate, dtype=np.float64)
    if reference.shape != candidate.shape or reference.shape != (len(case_ids),):
        raise ValueError("paired statistics inputs must have one value per case")
    patient_differences: dict[str, list[float]] = {}
    for case_id, difference in zip(case_ids, candidate - reference):
        patient_differences.setdefault(patient_by_case[case_id], []).append(float(difference))
    paired = np.asarray(
        [np.mean(patient_differences[patient]) for patient in sorted(patient_differences)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.choice(paired, size=(10000, len(paired)), replace=True).mean(axis=1)
    if bool(np.allclose(paired, 0.0)):
        p_value = 1.0
    else:
        p_value = float(stats.wilcoxon(paired, alternative="two-sided").pvalue)
    return {
        "mean_delta": float(paired.mean()),
        "patient_bootstrap_95_ci": [float(value) for value in np.percentile(draws, [2.5, 97.5])],
        "patient_wilcoxon_p": p_value,
        "patients": int(len(paired)),
        "patient_wins": int((paired > 0).sum()),
        "patient_ties": int(np.isclose(paired, 0.0).sum()),
        "patient_losses": int((paired < 0).sum()),
    }


def rollout_policy(policy, ep: RecoveryEpisode, *, sample: bool, device="cpu"):
    """Process candidates (P-desc); click/skip each; return (final_dice, logps, rewards)."""
    union = ep.seed_union.copy()
    d = _dice(union, ep.gt)
    logps, rewards, acts = [], [], []
    n = len(ep.masks)
    for i in range(n):
        state = np.concatenate([ep.feats[i], [ep.P[i], ep.vac[i], i / max(1, n), union.mean() * 50]]).astype(np.float32)
        logit = policy(torch.from_numpy(state).to(device))
        prob = torch.sigmoid(logit)
        if sample:
            a = int(torch.bernoulli(prob).item())
        else:
            a = int(prob.item() > 0.5)
        lp = torch.log(prob + 1e-8) if a == 1 else torch.log(1 - prob + 1e-8)
        r = 0.0
        if a == 1:
            nu = union | ep.masks[i]
            nd = _dice(nu, ep.gt)
            r = (nd - d) - 0.002        # dDice minus small click cost
            union = nu; d = nd
        logps.append(lp); rewards.append(r); acts.append(a)
    return d, logps, rewards


def _state_vec(ep, i, union):
    n = len(ep.masks)
    return np.concatenate([ep.feats[i], [ep.P[i], ep.vac[i], i / max(1, n), union.mean() * 50]]).astype(np.float32)


def _bc_warmstart(policy, train_eps, device, *, steps=300, lr=5e-3, p_thr=0.5):
    """Clone the greedy classifier (click iff P>thr) so REINFORCE starts useful."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    S, Y = [], []
    for ep in train_eps:
        u = ep.seed_union.copy()
        for i in range(len(ep.masks)):
            S.append(_state_vec(ep, i, u)); Y.append(1.0 if ep.P[i] > p_thr else 0.0)
            if ep.P[i] > p_thr:
                u = u | ep.masks[i]
    if not S:
        return
    S = torch.from_numpy(np.array(S)).to(device); Y = torch.from_numpy(np.array(Y, np.float32)).to(device)
    lossf = nn.BCEWithLogitsLoss()
    for _ in range(steps):
        opt.zero_grad(); loss = lossf(policy(S), Y); loss.backward(); opt.step()


def train_rl(train_eps, *, epochs=40, lr=1e-3, batch=8, gamma=0.99, seed=0, device="cpu", warmstart=True):
    set_seed(seed)
    policy = RecoveryPolicy().to(device)
    if warmstart:
        _bc_warmstart(policy, train_eps, device)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    for ep_i in range(epochs):
        order = rng.permutation(len(train_eps))
        for b0 in range(0, len(order), batch):
            batch_idx = order[b0:b0 + batch]
            loss = 0.0; ndeltas = []
            for j in batch_idx:
                ep = train_eps[j]
                if not ep.masks:
                    continue
                d, logps, rewards = rollout_policy(policy, ep, sample=True, device=device)
                R = 0.0; returns = []
                for r in reversed(rewards):
                    R = r + gamma * R; returns.insert(0, R)
                returns = torch.tensor(returns, dtype=torch.float32, device=device)
                if returns.numel() > 1:
                    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
                loss = loss + sum(-lp * G for lp, G in zip(logps, returns))
                ndeltas.append(d - _dice(ep.seed_union, ep.gt))
            if isinstance(loss, float):
                continue
            opt.zero_grad(); loss.backward(); opt.step()
        if ep_i % 5 == 0:
            mean_d = np.mean([rollout_policy(policy, e, sample=False, device=device)[0] - _dice(e.seed_union, e.gt)
                              for e in train_eps if e.masks])
            print(f"[rl-recovery] ep{ep_i} train mean dDice(seed)={mean_d:+.4f}", flush=True)
    return policy


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def _seg_local(sess, pet, coord, roi=36):
    b = _roi_bounds(pet.shape, coord, (roi, roi, roi)); sl = tuple(slice(lo, hi) for lo, hi in b)
    pr = pet[sl]; sd = tuple(coord[a] - b[a][0] for a in range(3))
    sess.reset_interactions(); sess.set_image(pr[None]); sess.set_target_buffer(torch.zeros(pr.shape, dtype=torch.uint8))
    m = _add_point(sess, sd, positive=True).astype(bool)
    o = np.zeros(pet.shape, bool); o[sl] = m
    if torch.cuda.is_available():           # release nnInteractive's per-call buffers to avoid fragmentation OOM
        torch.cuda.empty_cache()
    return o


def main(argv=None):
    import argparse, pickle
    from .autopet_pipeline import find_autopet_cases, lesion_components
    from .real_rollout import make_session, _add_point as _ap, _roi_bounds as _rb, DEFAULT_MODEL_ROOT
    global _add_point, _roi_bounds
    _add_point, _roi_bounds = _ap, _rb

    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--pct", type=float, default=97.0)
    p.add_argument("--roi", type=int, default=36)
    p.add_argument("--rl-epochs", type=int, default=60)
    p.add_argument("--focal-cap", type=int, default=250, help="reject candidate segs larger than this (organ over-seg)")
    p.add_argument("--grid", default="256,128,128", help="whole-body resample grid Z,Y,X (finer = more lesion recall)")
    p.add_argument("--min-size", type=int, default=4, help="min candidate component size")
    p.add_argument("--cache", default="")
    p.add_argument("--manifest", default="", help="StudyManifest v2 controlling patient-level splits")
    p.add_argument("--train-split", default="train")
    p.add_argument("--eval-split", default="test")
    p.add_argument("--checkpoint-sha256", default="", help="required when --cache is used")
    p.add_argument("--dataset-sha256", default="", help="required when --cache is used")
    p.add_argument("--target-label", default="TTB", help="semantic target bound into the cache key")
    args = p.parse_args(argv)
    GRID = tuple(int(x) for x in args.grid.split(","))
    # scale the focal cap + ROI with resolution so physical sizes stay comparable
    vox_ratio = float(np.prod(GRID)) / float(256 * 128 * 128)
    focal_cap = int(args.focal_cap * vox_ratio)
    roi = int(round(args.roi * (GRID[0] / 256.0)))

    def _pack(ep):
        return {"seed": np.argwhere(ep.seed_union).astype(np.int16), "shape": ep.seed_union.shape,
                "gt": np.argwhere(ep.gt).astype(np.int16), "feats": ep.feats, "P": ep.P, "vac": ep.vac,
                "masks": [np.argwhere(m).astype(np.int16) for m in ep.masks]}

    def _rec(coords, shape):
        a = np.zeros(shape, bool)
        if len(coords):
            a[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        return a

    def _unpack(d):
        return RecoveryEpisode(_rec(d["seed"], d["shape"]), _rec(d["gt"], d["shape"]),
                               d["feats"], d["P"], d["vac"], [_rec(m, d["shape"]) for m in d["masks"]])

    discovered_cases = find_autopet_cases(args.data_root, tracer="ANY")
    dataset_sha256 = args.dataset_sha256
    if args.manifest:
        observed_manifest_sha256 = sha256_file(Path(args.manifest))
        if dataset_sha256 and dataset_sha256 != observed_manifest_sha256:
            raise ValueError(
                f"--dataset-sha256 does not match --manifest: "
                f"expected {dataset_sha256}, observed {observed_manifest_sha256}"
            )
        dataset_sha256 = observed_manifest_sha256
        train_cases, val_cases, patient_by_case = _manifest_split_cases(
            discovered_cases,
            Path(args.manifest),
            train_split=args.train_split,
            eval_split=args.eval_split,
        )
        cases = train_cases + val_cases
    else:
        n_val = max(6, len(discovered_cases) // 5)
        val_cases, train_cases = discovered_cases[:n_val], discovered_cases[n_val:]
        cases = train_cases + val_cases
        patient_by_case = {case.case_id: case.case_id.rsplit("_", 1)[0] for case in cases}
    cache_identity = None
    if args.cache:
        cache_identity = CacheIdentity(
            namespace="autopet_rl_recovery_episodes",
            case_ids=tuple(case.case_id for case in cases),
            target_label=args.target_label,
            checkpoint_sha256=args.checkpoint_sha256,
            dataset_sha256=dataset_sha256,
            config_sha256=sha256_json(
                {
                    "grid_zyx": GRID,
                    "pct": args.pct,
                    "roi": roi,
                    "focal_cap": focal_cap,
                    "min_size": args.min_size,
                    "candidate_classifier_epochs": 300,
                    "seed_strategy": "highest_evidential_nonempty_candidate",
                    "train_split": args.train_split if args.manifest else "legacy-tail",
                    "eval_split": args.eval_split if args.manifest else "legacy-head",
                }
            ),
        )

    def build_clf(cs):
        F, L = [], []
        for c in cs:
            v = load_autopet_volume(c, target=GRID); ts = load_totseg(c, GRID)
            for cd in extract_candidates(v, ts, pet_pct=args.pct, min_size=args.min_size):
                F.append(cd.features); L.append(cd.label)
        return np.array(F), np.array(L)

    print("[rl-recovery] training candidate classifier ...", flush=True)
    trf, trl = build_clf(train_cases)
    clf = train_classifier(trf, trl, epochs=300)
    print(f"[rl-recovery] classifier lesion-rate {trl.mean():.3f} (n={len(trl)})", flush=True)

    sess = make_session(device="cuda", model_root=DEFAULT_MODEL_ROOT)

    def precompute(cs):
        eps = []
        for case_index, c in enumerate(cs, start=1):
            print(
                f"[rl-recovery] episode {case_index}/{len(cs)}: {c.case_id}",
                flush=True,
            )
            v = load_autopet_volume(c, target=GRID); ts = load_totseg(c, GRID)
            cands = extract_candidates(
                v,
                ts,
                pet_pct=args.pct,
                min_size=args.min_size,
                label_candidates=False,
            )
            if not cands:
                eps.append(RecoveryEpisode(np.zeros_like(v.gt), v.gt, np.zeros((0, FEAT_DIM), np.float32),
                                           np.zeros(0), np.zeros(0), [])); continue
            feats = np.array([cd.features for cd in cands]); P, vac = classify(clf, feats)
            order = np.argsort(P)[::-1]
            masks = []
            for i in order:
                m = _seg_local(sess, v.pet, cands[i].coord, roi=roi)
                # focal gate: an over-sized seg is physiological organ over-seg -> neutralize it
                masks.append(m if int(m.sum()) <= focal_cap else np.zeros_like(m))
            seed_index = next((index for index, mask in enumerate(masks) if bool(mask.any())), None)
            if seed_index is None:
                seed_union = np.zeros_like(v.gt)
                remaining = list(range(len(masks)))
            else:
                seed_union = masks[seed_index].copy()
                remaining = [index for index in range(len(masks)) if index != seed_index]
            eps.append(
                RecoveryEpisode(
                    seed_union,
                    v.gt,
                    feats[order][remaining],
                    P[order][remaining],
                    vac[order][remaining],
                    [masks[index] for index in remaining],
                )
            )
        return eps

    import pickle
    if args.cache and Path(args.cache).exists():
        print(f"[rl-recovery] loading cached episodes {args.cache}", flush=True)
        with Path(args.cache).open("rb") as handle:
            blob = unwrap_cache_envelope(pickle.load(handle), cache_identity)
        train_eps = [_unpack(d) for d in blob["train"]]; val_eps = [_unpack(d) for d in blob["val"]]
        val_cases = val_cases[: len(val_eps)]
    else:
        print("[rl-recovery] precomputing episodes (nnInteractive per candidate) ...", flush=True)
        train_eps = precompute(train_cases); val_eps = precompute(val_cases)
        if args.cache:
            cache_path = Path(args.cache)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"train": [_pack(e) for e in train_eps], "val": [_pack(e) for e in val_eps]}
            with cache_path.open("wb") as handle:
                pickle.dump(make_cache_envelope(cache_identity, payload), handle)
            print(f"[rl-recovery] cached episodes -> {args.cache}", flush=True)
    print(f"[rl-recovery] episodes: train {len(train_eps)} val {len(val_eps)}", flush=True)

    def base(ep, mode, thr=0.6):
        u = ep.seed_union.copy()
        if mode == "naive":
            for m in ep.masks:
                u = u | m
        elif mode == "greedy":
            for pp, m in zip(ep.P, ep.masks):
                if pp < thr:
                    break
                u = u | m
        return _dice(u, ep.gt)

    seed = [_dice(e.seed_union, e.gt) for e in val_eps]
    naive = [base(e, "naive") for e in val_eps]
    greedy = [base(e, "greedy") for e in val_eps]
    oracle = []
    for e in val_eps:
        u = e.seed_union.copy()
        current = _dice(u, e.gt)
        for mask in e.masks:
            candidate_union = u | mask
            candidate_dice = _dice(candidate_union, e.gt)
            if candidate_dice > current:
                u, current = candidate_union, candidate_dice
        oracle.append(current)

    policy = train_rl(train_eps, epochs=args.rl_epochs, device="cpu")
    rl = [rollout_policy(policy, e, sample=False)[0] for e in val_eps]

    case_ids = [case.case_id for case in val_cases]
    per_case = [
        {
            "case_id": case_id,
            "patient_id": patient_by_case[case_id],
            "seed": float(seed_value),
            "greedy": float(greedy_value),
            "rl": float(rl_value),
            "oracle": float(oracle_value),
        }
        for case_id, seed_value, greedy_value, rl_value, oracle_value in zip(
            case_ids, seed, greedy, rl, oracle
        )
    ]
    summary = {
        "n": len(val_eps),
        "patients": len({patient_by_case[case_id] for case_id in case_ids}),
        "protocol": {
            "manifest": str(Path(args.manifest).resolve()) if args.manifest else None,
            "manifest_sha256": dataset_sha256,
            "train_split": args.train_split if args.manifest else "legacy-tail",
            "eval_split": args.eval_split if args.manifest else "legacy-head",
            "seed_strategy": "highest_evidential_nonempty_candidate",
            "inference_candidate_generation_uses_ground_truth": False,
        },
        "seed": float(np.mean(seed)),
        "naive": float(np.mean(naive)), "naive_delta": float(np.mean(naive) - np.mean(seed)),
        "greedy": float(np.mean(greedy)), "greedy_delta": float(np.mean(greedy) - np.mean(seed)),
        "rl": float(np.mean(rl)), "rl_delta": float(np.mean(rl) - np.mean(seed)),
        "oracle": float(np.mean(oracle)), "oracle_delta": float(np.mean(oracle) - np.mean(seed)),
        "paired_statistics": {
            "greedy_vs_seed": _paired_patient_statistics(
                case_ids, patient_by_case, seed, greedy
            ),
            "rl_vs_seed": _paired_patient_statistics(case_ids, patient_by_case, seed, rl),
            "rl_vs_greedy": _paired_patient_statistics(case_ids, patient_by_case, greedy, rl),
        },
        "per_case": per_case,
    }
    print("\n=== WHOLE-BODY Dice (val, GT-free) ===")
    for k in ("seed", "naive", "greedy", "rl", "oracle"):
        dk = summary.get(f"{k}_delta")
        print(f"  {k:20s} {summary[k]:.3f}" + (f" ({dk:+.3f})" if dk is not None else ""))
    import json
    Path(args.out_dir, "runs").mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, "checkpoints").mkdir(parents=True, exist_ok=True)
    Path(args.out_dir, "runs", "rl_recovery_summary.json").write_text(json.dumps(summary, indent=2))
    torch.save(policy.state_dict(), str(Path(args.out_dir, "checkpoints", "rl_recovery_policy.pt")))
    print("[rl-recovery] DONE", flush=True)
    return summary


if __name__ == "__main__":
    main()
