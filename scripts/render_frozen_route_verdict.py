"""Render a compact PI-facing frozen route-policy verdict."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    policies = report["test_evaluation"]["policies"]

    policy_spec = [
        ("keep_resenc", "Keep\nResEnc", "#6b7280"),
        ("fixed_r2_intersection", "Fixed\nR2 intersection", "#60a5fa"),
        ("edl_accept_best_utility", "Learned\nEDL gate", "#ef4444"),
        ("linear_contextual_bandit", "Contextual\nbandit", "#f59e0b"),
        ("hindsight_oracle", "Hindsight\noracle", "#10b981"),
    ]
    means = [policies[key]["patient_estimand"]["mean_final_dice"] for key, _, _ in policy_spec]
    baseline = policies["keep_resenc"]["patient_estimand"]["mean_final_dice"]

    rows = report["test_evaluation"]["per_study"]
    cases = sorted({row["case_id"] for row in rows})
    by_key = {(row["case_id"], row["policy"]): row for row in rows}
    oracle_delta = [by_key[(case, "hindsight_oracle")]["delta_dice"] for case in cases]
    edl_delta = [by_key[(case, "edl_accept_best_utility")]["delta_dice"] for case in cases]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.2), constrained_layout=True)
    fig.patch.set_facecolor("#f8fafc")
    for axis in axes:
        axis.set_facecolor("white")
        axis.spines[["top", "right"]].set_visible(False)

    x = np.arange(len(policy_spec))
    colors = [color for _, _, color in policy_spec]
    axes[0].bar(x, means, color=colors, width=0.72)
    axes[0].axhline(baseline, color="#111827", linewidth=1.2, linestyle="--", label="ResEnc baseline")
    axes[0].set_xticks(x, [label for _, label, _ in policy_spec])
    axes[0].set_ylim(0.40, 0.70)
    axes[0].set_ylabel("Mean patient-level Dice")
    axes[0].set_title("Frozen internal test: learned gate did not generalize", loc="left", weight="bold")
    for index, value in enumerate(means):
        delta = value - baseline
        axes[0].text(index, value + 0.008, f"{value:.3f}\n({delta:+.3f})", ha="center", va="bottom", fontsize=9)

    width = 0.36
    x2 = np.arange(len(cases))
    axes[1].bar(x2 - width / 2, edl_delta, width, color="#ef4444", label="Learned EDL")
    axes[1].bar(x2 + width / 2, oracle_delta, width, color="#10b981", label="Hindsight oracle")
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[1].set_xticks(x2, [case.replace("train_", "") for case in cases], rotation=25, ha="right")
    axes[1].set_ylabel("Dice change vs keep ResEnc")
    axes[1].set_title("Action headroom exists, but EDL selected harmful replacements", loc="left", weight="bold")
    axes[1].legend(frameon=False, loc="lower right")

    fig.suptitle("AutoPET ResEnc + prompt-fusion route verdict", fontsize=17, weight="bold", x=0.02, ha="left")
    fig.text(
        0.02,
        0.01,
        "Exploratory only: 2 prior-exposed test patients / 4 studies, GT-derived robot-user prompts. "
        "Oracle +0.0346 Dice; learned EDL -0.1591; contextual bandit safely kept all masks.",
        fontsize=10,
        color="#374151",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
