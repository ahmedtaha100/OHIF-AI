"""Render a compact PI figure for AutoPET prompt-response trajectories."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    args = _parse_args()
    with args.csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("trajectory CSV is empty")

    x = np.arange(3)
    dice = np.asarray(
        [[float(row[f"dice_{key}"]) for key in ("zero", "round1", "round2")] for row in rows]
    )
    nsd = np.asarray(
        [[float(row[f"nsd_{key}"]) for key in ("zero", "round1", "round2")] for row in rows]
    )
    colors = ("#00a6fb", "#f15bb5", "#00b894", "#f4a261")
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True, facecolor="white")

    for axis, values, label in ((axes[0], dice, "Dice"), (axes[1], nsd, "NSD at 2 mm")):
        for row, trajectory, color in zip(rows, values, colors, strict=False):
            axis.plot(x, trajectory, marker="o", linewidth=1.8, color=color, alpha=0.8, label=row["case"])
        mean = values.mean(axis=0)
        axis.plot(x, mean, marker="o", linewidth=4, color="black", label="Mean")
        for xi, value in zip(x, mean, strict=True):
            axis.annotate(f"{value:.3f}", (xi, value), xytext=(0, 10), textcoords="offset points", ha="center", fontweight="bold")
        axis.set_xticks(x, ("0 corrections", "Round 1", "Round 2"))
        axis.set_ylim(-0.03, 1.0)
        axis.set_ylabel(label)
        axis.grid(alpha=0.22)
        axis.set_title(f"{label} trajectory", fontweight="bold")
    axes[0].legend(fontsize=8, loc="lower right")

    oracle_dice = dice.max(axis=1).mean()
    oracle_nsd = nsd.max(axis=1).mean()
    dice_means = list(dice.mean(axis=0)) + [oracle_dice]
    bars = axes[2].bar(
        np.arange(4),
        dice_means,
        color=("#adb5bd", "#52b788", "#40916c", "#6c5ce7"),
    )
    axes[2].set_xticks(np.arange(4), ("Zero", "Always R1", "Always R2", "Best STOP"))
    axes[2].set_ylim(0, 0.8)
    axes[2].set_ylabel("Mean Dice")
    axes[2].set_title("Accept/STOP headroom", fontweight="bold")
    axes[2].grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, dice_means, strict=True):
        axes[2].text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontweight="bold")
    axes[2].text(
        0.5,
        0.16,
        f"STOP oracle: +{oracle_dice - dice.mean(axis=0)[2]:.3f} Dice vs always R2\n"
        f"Best NSD: {oracle_nsd:.3f}",
        transform=axes[2].transAxes,
        ha="center",
        va="center",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f1f3f5", "edgecolor": "#6c5ce7"},
    )

    fig.suptitle(
        "Official AutoPET V prompt model — corrections help, unconditional extra rounds can hurt\n"
        "Exploratory only: 2 previously exposed patients / 4 paired FDG+PSMA studies, RTX 4080",
        fontsize=15,
        fontweight="bold",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, facecolor="white")
    plt.close(fig)
    print(args.output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


if __name__ == "__main__":
    main()
