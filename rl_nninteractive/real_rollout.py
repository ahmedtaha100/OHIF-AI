"""Real nnInteractive rollout on real tumor CT — the segmentation-improvement test.

Drives the *real* nnInteractive checkpoint in-process on the RTX 4080 and asks
the question the mock adapter cannot answer: **when the evidential (GT-free)
policy places prompts into the real segmentator, does the Dice actually climb,
and how does it compare to random clicks and to the ground-truth oracle?**

For each case it runs three prompt policies from the same initial in-tumor seed:
  - ``edl``    : next click from the evidential error model (no GT at inference)
  - ``random`` : a random voxel in the ROI (polarity from the current mask)
  - ``oracle`` : the GT FP/FN largest-error heuristic (upper bound)
and records Dice-vs-#clicks plus the wall-clock per nnInteractive prompt (the
throughput number that decides whether a rented GPU is needed).

Everything runs in a lesion-centred ROI (default 128^3) so the volume matches
the evidential model's training scale and stays fast. The ROI is centred on the
initial seed click (what a clinician places first), not on the full GT mask.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise ImportError("real_rollout requires PyTorch") from exc

from .evidential import predict_error_maps
from .evidential_candidates import evidential_next_action, evidential_stop_decision
from .evidential_eval import load_evidential_model
from .metrics import dice_score, noc_at_threshold
from .robot_user import largest_component_robot_action
from .env import POINT_POSITIVE, POINT_NEGATIVE, STOP

DEFAULT_MODEL_ROOT = "artifacts/rl_nninteractive/checkpoints"
DEFAULT_MODEL_NAME = "nnInteractive_v1.0"


def _window_ct(volume: np.ndarray, lo: float = -1000.0, hi: float = 400.0) -> np.ndarray:
    v = np.clip(volume.astype(np.float32), lo, hi)
    return ((v - lo) / (hi - lo) * 2.0 - 1.0).astype(np.float32)


def load_case_zyx(image_path: str | Path, label_path: str | Path, *, tumor_label: int = 1):
    """Return (raw_zyx float32 HU, windowed_zyx [-1,1], gt_zyx bool) in one axis order."""

    from .medical_geometry import load_nifti_on_reference_grid

    image_aligned = load_nifti_on_reference_grid(
        image_path,
        reference_path=image_path,
        channel_index=0,
        reference_channel_index=0,
    )
    label_aligned = load_nifti_on_reference_grid(
        label_path,
        reference_path=image_path,
        is_label=True,
        reference_channel_index=0,
    )
    raw = image_aligned.data_zyx.astype(np.float32, copy=False)
    gt = label_aligned.data_zyx == tumor_label
    return raw, _window_ct(raw), gt.astype(bool)


def _representative_coord(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.argwhere(mask)
    centroid = coords.mean(axis=0)
    d = np.square(coords - centroid).sum(axis=1)
    order = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0], d))
    c = coords[int(order[0])]
    return (int(c[0]), int(c[1]), int(c[2]))


def _roi_bounds(shape, center, patch):
    out = []
    for a in range(3):
        size = min(patch[a], shape[a])
        lo = int(np.clip(center[a] - size // 2, 0, shape[a] - size))
        out.append((lo, lo + size))
    return out


def make_session(*, device: str, model_root: str, model_name: str = DEFAULT_MODEL_NAME):
    from nnInteractive.inference.inference_session import nnInteractiveInferenceSession

    model_dir = Path(model_root) / model_name
    if not (model_dir / "fold_0" / "checkpoint_final.pth").exists():
        raise FileNotFoundError(f"nnInteractive checkpoint not found under {model_dir}")
    session = nnInteractiveInferenceSession(
        device=torch.device(device),
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=8,
        do_autozoom=True,
    )
    session.initialize_from_trained_model_folder(str(model_dir))
    return session


@dataclass
class PolicyRollout:
    policy: str
    dice_by_step: list[float] = field(default_factory=list)   # dice after each click (incl. seed at index 0)
    stop_step: int | None = None
    prompt_seconds: list[float] = field(default_factory=list)


def _add_point(session, coord, *, positive: bool) -> np.ndarray:
    session.add_point_interaction(tuple(int(c) for c in coord), include_interaction=positive)
    buf = session.target_buffer
    arr = buf.detach().cpu().numpy() if hasattr(buf, "detach") else np.asarray(buf)
    return arr.astype(np.uint8)


def run_policy(
    session,
    *,
    policy: str,
    roi_raw: np.ndarray,
    roi_win: np.ndarray,
    roi_gt: np.ndarray,
    seed: tuple[int, int, int],
    edl_model=None,
    device: str = "cuda",
    steps: int = 8,
    threshold: float = 0.30,
    min_size: int = 3,
    stop_error_voxels: int = 8,
    rng: np.random.Generator | None = None,
) -> PolicyRollout:
    rng = rng or np.random.default_rng(0)
    session.reset_interactions()
    session.set_image(roi_raw[None])
    session.set_target_buffer(torch.zeros(roi_raw.shape, dtype=torch.uint8))

    out = PolicyRollout(policy=policy)
    t0 = time.time()
    mask = _add_point(session, seed, positive=True)          # initial clinician seed
    out.prompt_seconds.append(time.time() - t0)
    out.dice_by_step.append(float(dice_score(mask.astype(bool), roi_gt)))

    for _ in range(steps):
        if policy == "oracle":
            dec = largest_component_robot_action(mask, roi_gt)
            if dec.action_type == STOP:
                out.stop_step = len(out.dice_by_step) - 1
                break
            coord, positive = dec.coord, dec.action_type == POINT_POSITIVE
        elif policy == "edl":
            maps = predict_error_maps(edl_model, roi_win, mask.astype(bool), device=device)
            stop = evidential_stop_decision(maps, mask.astype(bool), threshold=threshold,
                                            min_size=min_size, stop_error_voxels=stop_error_voxels)
            act = evidential_next_action(maps, mask.astype(bool), threshold=threshold, min_size=min_size)
            if stop.should_stop or act is None:
                out.stop_step = len(out.dice_by_step) - 1
                break
            coord, positive = act.coord, act.action_type == POINT_POSITIVE
        elif policy == "random":
            coord = tuple(int(rng.integers(0, s)) for s in roi_gt.shape)
            positive = not bool(mask[coord])
        else:
            raise ValueError(policy)

        t0 = time.time()
        mask = _add_point(session, coord, positive=positive)
        out.prompt_seconds.append(time.time() - t0)
        out.dice_by_step.append(float(dice_score(mask.astype(bool), roi_gt)))
    return out


@dataclass
class CaseResult:
    case_id: str
    seed_dice: float
    roi_shape: list[int]
    tumor_voxels: int
    policies: dict[str, dict[str, Any]]


def run_case(session, edl_model, image_path, label_path, *, device, patch, steps, seed_idx, threshold, min_size, stop_error_voxels=8, tumor_label=1):
    raw, win, gt = load_case_zyx(image_path, label_path, tumor_label=tumor_label)
    if not gt.any():
        return None
    seed_full = _representative_coord(gt)
    b = _roi_bounds(gt.shape, seed_full, (patch, patch, patch))
    sl = tuple(slice(lo, hi) for lo, hi in b)
    roi_raw, roi_win, roi_gt = raw[sl], win[sl], gt[sl]
    seed = tuple(seed_full[a] - b[a][0] for a in range(3))
    rng = np.random.default_rng(seed_idx)

    policies: dict[str, PolicyRollout] = {}
    rng_offset = {"edl": 1, "random": 2, "oracle": 3}
    for name in ("edl", "random", "oracle"):
        policies[name] = run_policy(
            session, policy=name, roi_raw=roi_raw, roi_win=roi_win, roi_gt=roi_gt, seed=seed,
            edl_model=edl_model, device=device, steps=steps, threshold=threshold, min_size=min_size,
            stop_error_voxels=stop_error_voxels,
            rng=np.random.default_rng(seed_idx * 10 + rng_offset[name]),
        )

    def pack(r: PolicyRollout) -> dict[str, Any]:
        noc85 = noc_at_threshold(r.dice_by_step[1:], 0.85) if len(r.dice_by_step) > 1 else None
        return {
            "dice_by_step": [round(d, 4) for d in r.dice_by_step],
            "final_dice": round(r.dice_by_step[-1], 4),
            "best_dice": round(max(r.dice_by_step), 4),
            "delta_vs_seed": round(r.dice_by_step[-1] - r.dice_by_step[0], 4),
            "stop_step": r.stop_step,
            "noc@85": noc85,
            "mean_prompt_sec": round(float(np.mean(r.prompt_seconds)), 3),
        }

    return CaseResult(
        case_id=Path(image_path).name,
        seed_dice=round(policies["edl"].dice_by_step[0], 4),
        roi_shape=list(roi_raw.shape),
        tumor_voxels=int(roi_gt.sum()),
        policies={k: pack(v) for k, v in policies.items()},
    )


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description="Real nnInteractive rollout: EDL vs random vs oracle")
    p.add_argument("--ckpt", required=True, help="trained evidential model checkpoint")
    p.add_argument("--msd-root", required=True)
    p.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    p.add_argument("--n-cases", type=int, default=6)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--patch", type=int, default=128)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--min-size", type=int, default=3)
    p.add_argument("--stop-error-voxels", type=int, default=8)
    p.add_argument("--tumor-label", type=int, default=1, help="MSD tumor label id (Pancreas cancer=2)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)

    from .evidential_dataset import find_msd_cases

    pairs = find_msd_cases(args.msd_root)
    # evaluate on the held-out split (first 1/5), matching training's val split
    n_val = max(1, len(pairs) // 5)
    held = pairs[:n_val][: args.n_cases]

    edl_model = load_evidential_model(args.ckpt, device=args.device)
    print(f"[rollout] loading nnInteractive ...", flush=True)
    session = make_session(device=args.device, model_root=args.model_root)
    print(f"[rollout] nnInteractive ready; {len(held)} held-out cases", flush=True)

    import faulthandler, traceback
    faulthandler.enable()
    results: list[CaseResult] = []
    for i, (img, lab) in enumerate(held):
        try:
            r = run_case(session, edl_model, img, lab, device=args.device, patch=args.patch,
                         steps=args.steps, seed_idx=1000 + i, threshold=args.threshold, min_size=args.min_size,
                         stop_error_voxels=args.stop_error_voxels, tumor_label=args.tumor_label)
        except Exception:
            print(f"[rollout] case {Path(img).name} ERROR:", flush=True)
            traceback.print_exc()
            continue
        if r is None:
            continue
        results.append(r)
        e, rnd, o = r.policies["edl"], r.policies["random"], r.policies["oracle"]
        print(f"[rollout] {r.case_id}: seed_dice={r.seed_dice:.3f} | "
              f"EDL {e['final_dice']:.3f} (d{e['delta_vs_seed']:+.3f}) | "
              f"random {rnd['final_dice']:.3f} (d{rnd['delta_vs_seed']:+.3f}) | "
              f"oracle {o['final_dice']:.3f} (d{o['delta_vs_seed']:+.3f})", flush=True)

    def agg(key: str, field_: str) -> float:
        vals = [r.policies[key][field_] for r in results if r.policies[key][field_] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    summary = {
        "n_cases": len(results),
        "steps": args.steps,
        "roi_patch": args.patch,
        "mean_seed_dice": round(float(np.mean([r.seed_dice for r in results])), 4) if results else None,
        "edl":    {"final": round(agg("edl", "final_dice"), 4), "delta": round(agg("edl", "delta_vs_seed"), 4), "prompt_sec": round(agg("edl", "mean_prompt_sec"), 3)},
        "random": {"final": round(agg("random", "final_dice"), 4), "delta": round(agg("random", "delta_vs_seed"), 4)},
        "oracle": {"final": round(agg("oracle", "final_dice"), 4), "delta": round(agg("oracle", "delta_vs_seed"), 4)},
        "cases": [asdict(r) for r in results],
    }
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== REAL nnInteractive rollout (held-out lung CT, ROI, mean over cases) ===")
    print(f"  seed Dice (1 click)        : {summary['mean_seed_dice']}")
    print(f"  EDL    final Dice / delta  : {summary['edl']['final']} / {summary['edl']['delta']:+}")
    print(f"  random final Dice / delta  : {summary['random']['final']} / {summary['random']['delta']:+}")
    print(f"  oracle final Dice / delta  : {summary['oracle']['final']} / {summary['oracle']['delta']:+}")
    print(f"  nnInteractive sec / prompt : {summary['edl']['prompt_sec']}  (real ROI volume, RTX 4080)")
    print(f"[report] {args.out}")
    return summary


if __name__ == "__main__":
    main()
