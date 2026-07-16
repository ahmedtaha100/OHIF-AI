"""Honest evaluation of the GT-free evidential policy against the GT oracle.

The claim being tested is precisely the one the prostate paper leaves open (and
that our own runner left blocked): *can the next correction be chosen without
ground truth?* So every number here is framed as an oracle gap -- the GT oracle
(``robot_user.largest_component_robot_action``) is the upper bound, and the
question is how close the evidential, GT-free policy gets to it.

Two levels:

Level 1 -- error-localization agreement (static, no environment):
    For held-out (image, current_mask, gt), compare where the GT oracle vs the
    evidential model vs a random baseline say the next correction should go.
    Reports polarity agreement, hit-rate on true errors, component precision,
    stop agreement, and uncertainty calibration (AUROC, ECE). This isolates the
    EDL contribution and uses no environment dynamics, so it is a real result on
    whatever data it is run on.

Level 2 -- interaction rollout (needs an nnInteractive-like session):
    ``compare_policies`` rolls the GT-oracle policy and the GT-free evidential
    policy through the same environment and reports Dice-at-stop, steps,
    NoC@85/90 and Dice@{1,3,5}. On the mock adapter this validates the wiring
    only (mock dynamics are synthetic); pass a real nnInteractive session to get
    a clinical rollout.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
except Exception as exc:  # pragma: no cover
    raise ImportError("evaluation requires PyTorch") from exc

from .deterministic_geometry import largest_error_component_mask
from .env import POINT_NEGATIVE, POINT_POSITIVE, STOP, RlNnInteractiveEnv
from .evidential import (
    EvidentialErrorNet3D,
    predict_error_maps,
    error_labels_from_masks,
)
from .evidential_candidates import (
    evidential_next_action,
    evidential_stop_decision,
)
from .robot_user import largest_component_robot_action
from .train_evidential import auroc, expected_calibration_error


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_evidential_model(
    ckpt_path: str | Path,
    *,
    device: "torch.device | str | None" = None,
) -> EvidentialErrorNet3D:
    dev = torch.device(device) if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(str(ckpt_path), map_location=dev, weights_only=False)
    model = EvidentialErrorNet3D(
        in_channels=int(ckpt.get("in_channels", 2)),
        base_channels=int(ckpt.get("base_channels", 16)),
    ).to(dev)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Level 1 -- localization agreement
# --------------------------------------------------------------------------- #
def _true_error_field(current_mask: np.ndarray, gt: np.ndarray, *, action_type: int) -> np.ndarray:
    mask = current_mask.astype(bool)
    target = gt.astype(bool)
    if action_type == POINT_POSITIVE:
        return np.logical_and(target, ~mask)   # false negatives
    if action_type == POINT_NEGATIVE:
        return np.logical_and(mask, ~target)   # false positives
    return np.zeros_like(mask)


@dataclass
class LocalizationAggregate:
    n_cases: int = 0
    n_oracle_acts: int = 0
    n_edl_acts: int = 0
    stop_agreement: float = 0.0
    polarity_agreement: float = 0.0            # among cases where both act
    edl_hit_rate: float = 0.0                  # EDL click lands on a true error of its polarity
    oracle_hit_rate: float = 0.0               # sanity: oracle click lands on a true error (~1.0)
    random_hit_rate: float = 0.0               # random click lands on a true error
    edl_in_oracle_component_rate: float = 0.0  # EDL click inside the oracle's chosen true component
    mean_component_precision: float = 0.0      # |edl_comp ∩ true_error| / |edl_comp|
    median_coord_distance: float = 0.0
    mean_auroc_perror: float = 0.0
    mean_ece: float = 0.0
    mean_current_dice: float = 0.0
    per_case: list[dict[str, Any]] = field(default_factory=list)


def evaluate_localization(
    model: EvidentialErrorNet3D,
    samples: Sequence[Any],
    *,
    device: "torch.device | str | None" = None,
    threshold: float = 0.30,
    min_size: int = 3,
    seed: int = 0,
    keep_per_case: bool = False,
) -> LocalizationAggregate:
    """Compare oracle / evidential / random next-correction on held-out samples.

    ``samples`` yields objects with ``.image``, ``.current_mask``, ``.gt``
    (e.g. ``EvidentialSample``).
    """

    rng = np.random.default_rng(seed)
    agg = LocalizationAggregate()
    stop_ok = pol_ok = pol_n = 0
    edl_hits = edl_n = 0
    oracle_hits = rand_hits = 0
    in_oracle = in_oracle_n = 0
    precisions: list[float] = []
    distances: list[float] = []
    aurocs: list[float] = []
    eces: list[float] = []
    dices: list[float] = []

    for s in samples:
        image = np.asarray(s.image, dtype=np.float32)
        mask = np.asarray(s.current_mask).astype(bool)
        gt = np.asarray(s.gt).astype(bool)
        inter = np.logical_and(mask, gt).sum()
        dice = 2.0 * inter / (mask.sum() + gt.sum() + 1e-8)
        dices.append(float(dice))

        oracle = largest_component_robot_action(mask, gt)
        maps = predict_error_maps(model, image, mask, device=device)
        edl = evidential_next_action(maps, mask, threshold=threshold, min_size=min_size)

        oracle_acts = oracle.action_type != STOP
        edl_acts = edl is not None
        agg.n_cases += 1
        agg.n_oracle_acts += int(oracle_acts)
        agg.n_edl_acts += int(edl_acts)
        stop_ok += int(oracle_acts == edl_acts)

        # calibration on this case
        is_err = error_labels_from_masks(mask, gt) != 0
        aurocs.append(auroc(maps.p_error, is_err))
        conf = maps.prob.max(axis=0)
        pred = maps.prob.argmax(axis=0)
        corr = pred == error_labels_from_masks(mask, gt)
        eces.append(expected_calibration_error(conf, corr))

        # random baseline click
        rand_coord = tuple(int(rng.integers(0, d)) for d in mask.shape)
        rand_polarity = POINT_NEGATIVE if mask[rand_coord] else POINT_POSITIVE
        rand_hits += int(_true_error_field(mask, gt, action_type=rand_polarity)[rand_coord])

        if oracle_acts:
            oracle_hits += int(_true_error_field(mask, gt, action_type=oracle.action_type)[oracle.coord])

        case: dict[str, Any] = {
            "case_id": getattr(s, "case_id", ""),
            "current_dice": float(dice),
            "oracle_acts": bool(oracle_acts),
            "edl_acts": bool(edl_acts),
            "oracle_polarity": int(oracle.action_type),
        }
        if oracle_acts and edl_acts:
            pol_n += 1
            match = int(edl.action_type == oracle.action_type)
            pol_ok += match
            edl_n += 1
            edl_true = _true_error_field(mask, gt, action_type=edl.action_type)
            hit = int(edl_true[edl.coord])
            edl_hits += hit
            precision = float(np.logical_and(edl.component_mask, edl_true).sum()) / float(edl.component_mask.sum())
            precisions.append(precision)
            dist = float(np.sqrt(sum((a - b) ** 2 for a, b in zip(edl.coord, oracle.coord))))
            distances.append(dist)
            # EDL click inside the oracle's chosen true component?
            oracle_comp = largest_error_component_mask(
                mask, gt, polarity="positive" if oracle.action_type == POINT_POSITIVE else "negative"
            )
            in_oracle_n += 1
            in_oracle += int(bool(oracle_comp[edl.coord]))
            case.update({
                "edl_polarity": int(edl.action_type),
                "polarity_match": bool(match),
                "edl_hit": bool(hit),
                "component_precision": precision,
                "coord_distance": dist,
                "edl_confidence": edl.mean_confidence,
                "edl_vacuity": edl.mean_vacuity,
            })
        if keep_per_case:
            agg.per_case.append(case)

    n = max(1, agg.n_cases)
    agg.stop_agreement = stop_ok / n
    agg.polarity_agreement = pol_ok / max(1, pol_n)
    agg.edl_hit_rate = edl_hits / max(1, edl_n)
    agg.oracle_hit_rate = oracle_hits / max(1, agg.n_oracle_acts)
    agg.random_hit_rate = rand_hits / n
    agg.edl_in_oracle_component_rate = in_oracle / max(1, in_oracle_n)
    agg.mean_component_precision = float(np.mean(precisions)) if precisions else 0.0
    agg.median_coord_distance = float(np.median(distances)) if distances else 0.0
    agg.mean_auroc_perror = float(np.nanmean(aurocs)) if aurocs else float("nan")
    agg.mean_ece = float(np.mean(eces)) if eces else float("nan")
    agg.mean_current_dice = float(np.mean(dices)) if dices else 0.0
    return agg


# --------------------------------------------------------------------------- #
# Level 2 -- interaction rollout
# --------------------------------------------------------------------------- #
@dataclass
class RolloutResult:
    policy: str
    final_dice: float
    steps: int
    dice_by_step: tuple[float, ...]
    stop_reason: str


def run_evidential_policy(
    env: RlNnInteractiveEnv,
    model: EvidentialErrorNet3D,
    *,
    image: np.ndarray,
    ground_truth: np.ndarray,
    initial_point: Sequence[int] | None = None,
    device: "torch.device | str | None" = None,
    threshold: float = 0.30,
    min_size: int = 3,
    stop_error_voxels: int = 8,
    max_steps: int | None = None,
) -> RolloutResult:
    """Roll the GT-free evidential policy through the env.

    The policy chooses actions from the evidential error map only; ground truth
    is used solely by the env to *measure* Dice (never by the policy).
    """

    options: dict[str, Any] = {"image": image, "ground_truth": np.asarray(ground_truth).astype(bool)}
    if initial_point is not None:
        options["initial_point"] = initial_point
    obs, info = env.reset(options=options)
    step_limit = env.max_interactions + 1 if max_steps is None else int(max_steps)
    dice_by_step: list[float] = []
    stop_reason = "max_steps"
    for _ in range(step_limit):
        mask = obs["mask"].astype(bool)
        maps = predict_error_maps(model, np.asarray(image, dtype=np.float32), mask, device=device)
        stop = evidential_stop_decision(
            maps, mask, threshold=threshold, min_size=min_size, stop_error_voxels=stop_error_voxels
        )
        action_cand = evidential_next_action(maps, mask, threshold=threshold, min_size=min_size)
        if stop.should_stop or action_cand is None:
            obs, reward, terminated, truncated, info = env.step({"action_type": STOP})
            stop_reason = f"edl_stop:{stop.reason}"
            break
        obs, reward, terminated, truncated, info = env.step(
            {"action_type": action_cand.action_type, "coord": action_cand.coord}
        )
        dice_by_step.append(float(info["dice"]))
        if terminated or truncated:
            stop_reason = str(info.get("done_reason", "env_done"))
            break
    return RolloutResult(
        policy="evidential_gt_free",
        final_dice=float(info["dice"]),
        steps=len(dice_by_step),
        dice_by_step=tuple(dice_by_step),
        stop_reason=stop_reason,
    )


def run_oracle_policy(
    env: RlNnInteractiveEnv,
    *,
    image: np.ndarray,
    ground_truth: np.ndarray,
    initial_point: Sequence[int] | None = None,
    max_steps: int | None = None,
) -> RolloutResult:
    """Roll the GT-oracle FP/FN policy (upper bound) through the same env."""

    from .robot_user import run_largest_component_robot_user

    episode = run_largest_component_robot_user(
        env, image=image, ground_truth=np.asarray(ground_truth).astype(bool),
        initial_point=initial_point, max_steps=max_steps,
    )
    return RolloutResult(
        policy="gt_oracle_fpfn",
        final_dice=float(episode.final_info["dice"]),
        steps=len([d for d in episode.decisions if d.action_type != STOP]),
        dice_by_step=episode.dice_by_step,
        stop_reason="oracle_no_error" if episode.terminated else "max_steps",
    )


def summarize_run(results: list[RolloutResult]) -> dict[str, float]:
    from .metrics import noc_at_threshold

    final = [r.final_dice for r in results]
    steps = [r.steps for r in results]
    def _noc(thr: float) -> float:
        # Mean number-of-clicks to reach the Dice threshold, over cases that reach it.
        vals = []
        for r in results:
            if r.dice_by_step:
                n = noc_at_threshold(r.dice_by_step, thr)
                if n is not None:
                    vals.append(int(n))
        return float(np.mean(vals)) if vals else float("nan")
    return {
        "mean_final_dice": float(np.mean(final)) if final else float("nan"),
        "mean_steps": float(np.mean(steps)) if steps else float("nan"),
        "noc@85": _noc(0.85),
        "noc@90": _noc(0.90),
        "n": len(results),
    }


def write_report(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


# --------------------------------------------------------------------------- #
# CLI: run the honest evaluation on a held-out split
# --------------------------------------------------------------------------- #
def _mock_rollout_comparison(model, samples, device, *, volume_shape) -> dict[str, Any]:
    """Wiring-only rollout on the mock adapter (synthetic dynamics, NOT clinical).

    Confirms the GT-free policy drives the env end-to-end and stops on its own.
    """

    from .mock_adapter import MockNnInteractiveSession

    oracle_results: list[RolloutResult] = []
    edl_results: list[RolloutResult] = []
    for s in samples:
        image = np.asarray(s.image, dtype=np.float32)[None, ...]
        gt = np.asarray(s.gt).astype(bool)
        coords = np.argwhere(gt)
        seed_pt = tuple(int(c) for c in coords[len(coords) // 2]) if len(coords) else None
        env_o = RlNnInteractiveEnv(volume_shape, max_interactions=8, session_factory=MockNnInteractiveSession)
        oracle_results.append(run_oracle_policy(env_o, image=image, ground_truth=gt, initial_point=seed_pt))
        env_e = RlNnInteractiveEnv(volume_shape, max_interactions=8, session_factory=MockNnInteractiveSession)
        edl_results.append(run_evidential_policy(env_e, model, image=image, ground_truth=gt,
                                                 initial_point=seed_pt, device=device))
    return {
        "note": "MOCK adapter dynamics are synthetic; this validates wiring/stop behavior, not clinical performance.",
        "gt_oracle_fpfn": summarize_run(oracle_results),
        "evidential_gt_free": summarize_run(edl_results),
    }


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    import argparse

    from .evidential_dataset import SyntheticEvidentialDataset

    p = argparse.ArgumentParser(description="Honest evaluation of the GT-free evidential policy")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", choices=["synthetic", "msd"], default="synthetic")
    p.add_argument("--msd-root", default="")
    p.add_argument("--samples-per-case", type=int, default=10)
    p.add_argument("--n", type=int, default=120)
    p.add_argument("--held-out-seed", type=int, default=20_000_000)
    p.add_argument("--threshold", type=float, default=0.30)
    p.add_argument("--min-size", type=int, default=3)
    p.add_argument("--device", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--mock-rollout", type=int, default=20, help="number of cases for the mock-adapter wiring rollout")
    args = p.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = load_evidential_model(args.ckpt, device=device)

    if args.data == "msd":
        # Evaluate on the held-out val split (cases NOT used for training), with a
        # fresh perturbation seed so the current-mask states are unseen too.
        from .evidential_dataset import find_msd_cases, materialize_msd_samples

        pairs = find_msd_cases(args.msd_root)
        n_val = max(1, len(pairs) // 5)
        held_out_pairs = pairs[:n_val]
        samples = materialize_msd_samples(
            held_out_pairs, base_seed=args.held_out_seed, samples_per_case=args.samples_per_case
        )
    else:
        ds = SyntheticEvidentialDataset(args.n, base_seed=args.held_out_seed)
        samples = [ds.make(i) for i in range(args.n)]

    loc = evaluate_localization(
        model, samples, device=device, threshold=args.threshold, min_size=args.min_size, keep_per_case=False
    )
    rollout = _mock_rollout_comparison(
        model, samples[: args.mock_rollout], device, volume_shape=samples[0].image.shape
    )
    payload = {
        "checkpoint": args.ckpt,
        "held_out_seed": args.held_out_seed,
        "n_cases": args.n,
        "threshold": args.threshold,
        "min_size": args.min_size,
        "localization": asdict(loc),
        "mock_rollout": rollout,
    }
    write_report(args.out, payload)

    print(f"=== GT-free evidential localization vs GT oracle (held-out {args.data}) ===")
    print(f"  cases                          : {loc.n_cases}")
    print(f"  mean current-mask Dice         : {loc.mean_current_dice:.3f}")
    print(f"  AUROC(p_error -> voxel error)  : {loc.mean_auroc_perror:.3f}")
    print(f"  ECE (Dirichlet mean)           : {loc.mean_ece:.3f}")
    print(f"  stop agreement w/ oracle       : {loc.stop_agreement:.3f}")
    print(f"  polarity agreement (FN vs FP)  : {loc.polarity_agreement:.3f}")
    print(f"  EDL next-click hits true error : {loc.edl_hit_rate:.3f}   (random={loc.random_hit_rate:.3f}, oracle={loc.oracle_hit_rate:.3f})")
    print(f"  EDL click in oracle component  : {loc.edl_in_oracle_component_rate:.3f}")
    print(f"  EDL component precision        : {loc.mean_component_precision:.3f}")
    print(f"  median |EDL-oracle| click dist : {loc.median_coord_distance:.2f} voxels")
    print("=== mock-adapter rollout (WIRING ONLY, synthetic dynamics) ===")
    print(f"  oracle : {rollout['gt_oracle_fpfn']}")
    print(f"  edl    : {rollout['evidential_gt_free']}")
    print(f"[report] {args.out}")
    return payload


if __name__ == "__main__":
    main()
