"""Reinforcement-learning interaction policy over the GT-free EDL substrate.

This is the actual RL policy (not the greedy heuristic): a neural policy trained
with policy-gradient (REINFORCE + a value baseline, warm-started by cloning the
greedy evidential policy) against the **real nnInteractive** environment.

At every step the evidential model proposes a small menu of GT-free candidate
corrections (top-k predicted false-negative / false-positive components) plus a
STOP action. The policy sees only those GT-free features and learns *which*
correction to apply and *when to stop*. Ground truth is used only to compute the
training reward (delta-Dice minus a per-click cost); the policy's inputs are
ground-truth-free, so the learned policy is deployable.

Why RL over the greedy policy: on strong nnInteractive the greedy policy never
stops and sometimes over-clicks (see rollout_final_v2_p64.json). The RL objective
(cumulative delta-Dice minus step cost, with STOP terminal) directly teaches the
policy to stop before a click would hurt and to prefer clicks that help — the
headroom between greedy (+0.023) and the GT oracle (+0.050).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except Exception as exc:  # pragma: no cover
    raise ImportError("rl_policy requires PyTorch") from exc

from .evidential import predict_error_maps
from .evidential_candidates import (
    evidential_candidates_topk,
    evidential_next_action,
    evidential_stop_decision,
)
from .evidential_eval import load_evidential_model
from .evidential_dataset import find_msd_cases
from .metrics import dice_score
from .robot_user import largest_component_robot_action
from .real_rollout import (
    DEFAULT_MODEL_ROOT, load_case_zyx, make_session, _roi_bounds, _representative_coord, _add_point,
)
from .env import POINT_POSITIVE, STOP

CAND_FEAT_DIM = 10
STATE_FEAT_DIM = 4
FEAT_DIM = CAND_FEAT_DIM + STATE_FEAT_DIM


# --------------------------------------------------------------------------- #
# Networks
# --------------------------------------------------------------------------- #
class PolicyNet(nn.Module):
    def __init__(self, in_dim: int = FEAT_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, cand_feats: "torch.Tensor") -> "torch.Tensor":
        return self.net(cand_feats).squeeze(-1)      # (n_candidates,)


class ValueNet(nn.Module):
    def __init__(self, in_dim: int = STATE_FEAT_DIM, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, state: "torch.Tensor") -> "torch.Tensor":
        return self.net(state).squeeze(-1)


# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
def _state_summary(maps, mask: np.ndarray, step_frac: float, threshold: float) -> np.ndarray:
    total_err = float((maps.p_error >= threshold).mean())
    fn = (maps.p_false_negative >= threshold) & (~mask)
    fp = (maps.p_false_positive >= threshold) & (mask)
    largest = 0.0
    for field in (fn, fp):
        if field.any():
            largest = max(largest, float(field.sum()) / mask.size)
    return np.array([total_err, largest, float(maps.vacuity.mean()), step_frac], dtype=np.float32)


def build_action_features(maps, mask: np.ndarray, cands: list, step_frac: float, threshold: float):
    """Return (feature matrix [n_actions, FEAT_DIM], state_summary [STATE_FEAT_DIM], actions).

    ``actions`` is ``cands + [None]`` where the trailing None is the STOP action.
    """

    state = _state_summary(maps, mask, step_frac, threshold)
    shape = np.asarray(mask.shape, dtype=np.float32)
    rows = []
    for c in cands:
        rows.append([
            0.0,
            1.0 if c.polarity == "positive" else 0.0,
            1.0 if c.polarity == "negative" else 0.0,
            c.component_size / mask.size,
            min(c.predicted_error_mass / 1000.0, 10.0),
            c.mean_confidence,
            c.mean_vacuity,
            c.coord[0] / shape[0], c.coord[1] / shape[1], c.coord[2] / shape[2],
        ])
    rows.append([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5])  # STOP
    cand_part = np.asarray(rows, dtype=np.float32)
    state_tile = np.tile(state, (cand_part.shape[0], 1))
    feats = np.concatenate([cand_part, state_tile], axis=1)
    return feats, state, cands + [None]


# --------------------------------------------------------------------------- #
# Environment wrapper (real nnInteractive, 64^3 ROI)
# --------------------------------------------------------------------------- #
class RealEdlEnv:
    def __init__(self, session, edl_model, *, device: str, patch: int = 64,
                 step_cost: float = 0.01, threshold: float = 0.4, min_size: int = 8, k: int = 3,
                 explore_coef: float = 0.0):
        self.session = session
        self.edl = edl_model
        self.device = device
        self.patch = patch
        self.step_cost = step_cost
        self.threshold = threshold
        self.min_size = min_size
        self.k = k
        # Yang et al. (IJCARS 2026) reward the policy for exploring uncertain regions;
        # their ablation shows this term carries the whole gain. Our GT-free analog:
        # reward reducing the EDL-predicted error mass (uncertainty resolved this step).
        self.explore_coef = explore_coef
        self._prev_err = 0.0

    def reset(self, raw, win, gt):
        seed_full = _representative_coord(gt)
        b = _roi_bounds(gt.shape, seed_full, (self.patch,) * 3)
        sl = tuple(slice(lo, hi) for lo, hi in b)
        self.roi_raw, self.roi_win, self.roi_gt = raw[sl], win[sl], gt[sl]
        self.seed = tuple(seed_full[a] - b[a][0] for a in range(3))
        self.session.reset_interactions()
        self.session.set_image(self.roi_raw[None])
        self.session.set_target_buffer(torch.zeros(self.roi_raw.shape, dtype=torch.uint8))
        self.mask = _add_point(self.session, self.seed, positive=True)
        self.dice = float(dice_score(self.mask.astype(bool), self.roi_gt))
        self.seed_dice = self.dice
        obs = self._observe(step_frac=0.0)
        self._prev_err = self._err_frac(obs["maps"])
        return obs

    def _err_frac(self, maps) -> float:
        return float((maps.p_error >= self.threshold).mean())

    def _observe(self, step_frac: float):
        maps = predict_error_maps(self.edl, self.roi_win, self.mask.astype(bool), device=self.device)
        cands = evidential_candidates_topk(maps, self.mask.astype(bool), k=self.k,
                                           threshold=self.threshold, min_size=self.min_size)
        feats, state, actions = build_action_features(maps, self.mask.astype(bool), cands, step_frac, self.threshold)
        return {"feats": feats, "state": state, "actions": actions, "maps": maps}

    def step(self, action, step_frac: float):
        if action is None:                    # STOP
            return None, 0.0, True
        positive = action.action_type == POINT_POSITIVE
        prev = self.dice
        self.mask = _add_point(self.session, action.coord, positive=positive)
        self.dice = float(dice_score(self.mask.astype(bool), self.roi_gt))
        obs = self._observe(step_frac=step_frac)
        new_err = self._err_frac(obs["maps"])
        # reward = Dice gain - click cost + explore bonus for resolving EDL-predicted error
        reward = (self.dice - prev) - self.step_cost + self.explore_coef * (self._prev_err - new_err)
        self._prev_err = new_err
        return obs, reward, False


# --------------------------------------------------------------------------- #
# Training (BC warm-start to greedy, then REINFORCE)
# --------------------------------------------------------------------------- #
def _greedy_teacher_index(env: RealEdlEnv, obs) -> int:
    """Index the greedy evidential policy would choose among obs['actions']."""

    mask = env.mask.astype(bool)
    stop = evidential_stop_decision(obs["maps"], mask, threshold=env.threshold,
                                    min_size=env.min_size, stop_error_voxels=20)
    act = evidential_next_action(obs["maps"], mask, threshold=env.threshold, min_size=env.min_size)
    if stop.should_stop or act is None:
        return len(obs["actions"]) - 1        # STOP is last
    # match greedy's chosen coord to a candidate
    for i, c in enumerate(obs["actions"][:-1]):
        if c is not None and c.coord == act.coord and c.polarity == act.polarity:
            return i
    # fallback: largest predicted-error-mass candidate
    masses = [c.predicted_error_mass for c in obs["actions"][:-1]]
    return int(np.argmax(masses)) if masses else len(obs["actions"]) - 1


@dataclass
class TrainConfig:
    bc_episodes: int = 200
    rl_episodes: int = 1200
    batch_episodes: int = 8
    max_steps: int = 8
    lr: float = 3e-4
    gamma: float = 0.99
    entropy_coef: float = 0.01
    explore_coef: float = 0.0
    seed: int = 0


def train(cases, session, edl_model, *, device: str, cfg: TrainConfig, out_ckpt: Path, log_every: int = 50, tumor_label: int = 1):
    torch.manual_seed(cfg.seed)
    policy = PolicyNet().to(device)
    value = ValueNet().to(device)
    opt = torch.optim.Adam(list(policy.parameters()) + list(value.parameters()), lr=cfg.lr)
    env = RealEdlEnv(session, edl_model, device=device, explore_coef=cfg.explore_coef)
    rng = np.random.default_rng(cfg.seed)
    loaded = [load_case_zyx(i, l, tumor_label=tumor_label) for i, l in cases]
    loaded = [(r, w, g) for (r, w, g) in loaded if g.any()]

    # ---- Phase A: behaviour-clone the greedy policy (cross-entropy to teacher) ----
    ce = nn.CrossEntropyLoss()
    for ep in range(cfg.bc_episodes):
        raw, win, gt = loaded[rng.integers(0, len(loaded))]
        obs = env.reset(raw, win, gt)
        opt.zero_grad()
        losses = []
        for t in range(cfg.max_steps):
            teacher = _greedy_teacher_index(env, obs)
            feats = torch.from_numpy(obs["feats"]).to(device)
            logits = policy(feats).unsqueeze(0)
            losses.append(ce(logits, torch.tensor([teacher], device=device)))
            action = obs["actions"][teacher]
            step = env.step(action, (t + 1) / cfg.max_steps)
            obs = step[0]
            if step[2] or obs is None:
                break
        if losses:
            torch.stack(losses).mean().backward()
            opt.step()
        if (ep + 1) % log_every == 0:
            print(f"[rl] BC ep{ep+1}/{cfg.bc_episodes}", flush=True)

    # ---- Phase B: REINFORCE with value baseline ----
    best = -1e9
    batch_logps, batch_advs, batch_vpreds, batch_vtargs, batch_ents = [], [], [], [], []
    ep_deltas = []
    for ep in range(cfg.rl_episodes):
        raw, win, gt = loaded[rng.integers(0, len(loaded))]
        obs = env.reset(raw, win, gt)
        logps, ents, states, rewards = [], [], [], []
        for t in range(cfg.max_steps):
            feats = torch.from_numpy(obs["feats"]).to(device)
            logits = policy(feats)
            dist = torch.distributions.Categorical(logits=logits)
            a = dist.sample()
            logps.append(dist.log_prob(a))
            ents.append(dist.entropy())
            states.append(torch.from_numpy(obs["state"]).to(device))
            action = obs["actions"][int(a)]
            nobs, reward, done = env.step(action, (t + 1) / cfg.max_steps)
            rewards.append(reward)
            obs = nobs
            if done or obs is None:
                break
        # returns + advantages
        R = 0.0
        returns = []
        for r in reversed(rewards):
            R = r + cfg.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
        vpreds = value(torch.stack(states))
        adv = (returns_t - vpreds).detach()
        for lp, e, vp, vt in zip(logps, ents, vpreds, returns_t):
            batch_logps.append(lp); batch_ents.append(e); batch_vpreds.append(vp); batch_vtargs.append(vt)
        batch_advs.append(adv)
        ep_deltas.append(env.dice - env.seed_dice)

        if (ep + 1) % cfg.batch_episodes == 0:
            advs = torch.cat(batch_advs)
            advs = (advs - advs.mean()) / (advs.std() + 1e-6)
            logp = torch.stack(batch_logps)
            ent = torch.stack(batch_ents).mean()
            vpred = torch.stack(batch_vpreds); vtarg = torch.stack(batch_vtargs)
            policy_loss = -(logp * advs).mean() - cfg.entropy_coef * ent
            value_loss = (vpred - vtarg).pow(2).mean()
            opt.zero_grad(); (policy_loss + 0.5 * value_loss).backward(); opt.step()
            batch_logps, batch_advs, batch_vpreds, batch_vtargs, batch_ents = [], [], [], [], []
            recent = float(np.mean(ep_deltas[-cfg.batch_episodes * 4:]))
            if (ep + 1) % log_every == 0:
                print(f"[rl] RL ep{ep+1}/{cfg.rl_episodes} mean_delta(recent)={recent:+.4f} "
                      f"pol_loss={float(policy_loss):.3f} val_loss={float(value_loss):.3f} ent={float(ent):.3f}", flush=True)
            if recent > best:
                best = recent
                torch.save({"policy": policy.state_dict(), "value": value.state_dict(),
                            "cfg": vars(cfg), "best_mean_delta": best}, out_ckpt)
    # always save final
    torch.save({"policy": policy.state_dict(), "value": value.state_dict(),
                "cfg": vars(cfg), "best_mean_delta": best, "final": True}, out_ckpt.with_suffix(".final.pt"))
    print(f"[rl] DONE best_mean_delta={best:+.4f} ckpt={out_ckpt}", flush=True)
    return policy, value


@torch.no_grad()
def run_rl_policy(env: RealEdlEnv, policy: PolicyNet, raw, win, gt, *, max_steps: int, device: str):
    obs = env.reset(raw, win, gt)
    seed = env.seed_dice
    for t in range(max_steps):
        feats = torch.from_numpy(obs["feats"]).to(device)
        a = int(torch.argmax(policy(feats)))
        action = obs["actions"][a]
        if action is None:
            break
        obs, _, done = env.step(action, (t + 1) / max_steps)
        if done or obs is None:
            break
    return seed, env.dice


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    p = argparse.ArgumentParser(description="Train the RL interaction policy on the EDL substrate")
    p.add_argument("--edl-ckpt", required=True)
    p.add_argument("--msd-root", required=True)
    p.add_argument("--model-root", default=DEFAULT_MODEL_ROOT)
    p.add_argument("--n-train-cases", type=int, default=40)
    p.add_argument("--tumor-label", type=int, default=1, help="MSD tumor label id (Pancreas cancer=2)")
    p.add_argument("--bc-episodes", type=int, default=200)
    p.add_argument("--rl-episodes", type=int, default=1200)
    p.add_argument("--explore-coef", type=float, default=0.0, help="Yang-et-al-style exploration reward weight")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--run-id", default="rl_policy_lung_v1")
    args = p.parse_args(argv)

    device = args.device if torch.cuda.is_available() else "cpu"
    pairs = find_msd_cases(args.msd_root)
    n_val = max(1, len(pairs) // 5)
    train_pairs, held = pairs[n_val:][: args.n_train_cases], pairs[:n_val]

    edl = load_evidential_model(args.edl_ckpt, device=device)
    print("[rl] loading nnInteractive ...", flush=True)
    session = make_session(device=device, model_root=args.model_root)
    ckpt_dir = Path(args.out_dir) / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = ckpt_dir / f"{args.run_id}_best.pt"

    cfg = TrainConfig(bc_episodes=args.bc_episodes, rl_episodes=args.rl_episodes, explore_coef=args.explore_coef)
    policy, _ = train(train_pairs, session, edl, device=device, cfg=cfg, out_ckpt=out_ckpt, tumor_label=args.tumor_label)

    # ---- held-out eval: RL vs greedy vs oracle vs random ----
    env = RealEdlEnv(session, edl, device=device)
    from .real_rollout import run_policy as rollout_run
    rl_d, seeds = [], []
    for img, lab in held:
        raw, win, gt = load_case_zyx(img, lab, tumor_label=args.tumor_label)
        if not gt.any():
            continue
        s, f = run_rl_policy(env, policy, raw, win, gt, max_steps=8, device=device)
        seeds.append(s); rl_d.append(f)
    summary = {
        "run_id": args.run_id,
        "held_out_cases": len(rl_d),
        "seed_dice": round(float(np.mean(seeds)), 4),
        "rl_final_dice": round(float(np.mean(rl_d)), 4),
        "rl_delta": round(float(np.mean(rl_d) - np.mean(seeds)), 4),
        "best_mean_delta_train": None,
    }
    (Path(args.out_dir) / "runs" / f"{args.run_id}_eval.json").parent.mkdir(parents=True, exist_ok=True)
    (Path(args.out_dir) / "runs" / f"{args.run_id}_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n=== RL policy held-out eval (GT-free at inference) ===")
    print(f"  cases          : {summary['held_out_cases']}")
    print(f"  seed Dice      : {summary['seed_dice']}")
    print(f"  RL final Dice  : {summary['rl_final_dice']}  (delta {summary['rl_delta']:+})")
    return summary


if __name__ == "__main__":
    main()
