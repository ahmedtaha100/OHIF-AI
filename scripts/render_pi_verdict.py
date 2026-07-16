#!/usr/bin/env python3
"""Render the frozen-split PI comparison from rl_recovery_summary.json."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    labels = ["Evidential\nseed", "Evidential\ngreedy", "RL +\nevidential", "Action-space\noracle"]
    keys = ["seed", "greedy", "rl", "oracle"]
    values = [float(summary[key]) for key in keys]
    colors = ["#78909C", "#42A5F5", "#7E57C2", "#26A69A"]

    rl_stats = summary["paired_statistics"]["rl_vs_seed"]
    greedy_stats = summary["paired_statistics"]["greedy_vs_seed"]
    rl_vs_greedy = summary["paired_statistics"]["rl_vs_greedy"]
    ci_low, ci_high = rl_stats["patient_bootstrap_95_ci"]
    if greedy_stats["mean_delta"] > 0 and rl_vs_greedy["mean_delta"] < 0:
        verdict = "EVIDENTIAL RECOVERY WORKED; RL DID NOT ADD VALUE"
    elif rl_stats["mean_delta"] > 0 and ci_low > 0 and rl_vs_greedy["mean_delta"] >= 0:
        verdict = "RL + EVIDENTIAL RECOVERY WORKED"
    elif rl_stats["mean_delta"] > 0 and rl_vs_greedy["mean_delta"] >= 0:
        verdict = "RL + EVIDENTIAL IS PROMISING, NOT CONFIRMED"
    else:
        verdict = "DID NOT IMPROVE THE FROZEN SPLIT"

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    bars = ax.bar(labels, values, color=colors, width=0.66)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.008,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )
    ax.set_ylabel("Whole-body Dice")
    ax.set_ylim(0, max(values) + 0.11)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"{verdict}\nFrozen patient-split RL + evidential-DL evaluation", fontweight="bold")
    ax.text(
        0.5,
        -0.22,
        (
            f"RL vs seed Δ={rl_stats['mean_delta']:+.3f}; patient-bootstrap 95% CI "
            f"[{ci_low:+.3f}, {ci_high:+.3f}]; Wilcoxon p={rl_stats['patient_wilcoxon_p']:.3f}; "
            f"RL vs greedy Δ={rl_vs_greedy['mean_delta']:+.3f}; "
            f"n={summary['patients']} patients / {summary['n']} studies"
        ),
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
