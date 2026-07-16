"""Render an honest PI-ready summary of a frozen route-policy report."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


POLICIES = {
    "oracle": "hindsight_oracle",
    "keep": "keep_resenc",
    "bandit": "linear_contextual_bandit",
    "fixed": "fixed_r2_intersection",
    "edl": "edl_accept_best_utility",
}

COLORS = {
    "ink": "#172033",
    "muted": "#677185",
    "grid": "#DDE2EA",
    "background": "#F7F8FA",
    "panel": "#FFFFFF",
    "oracle": "#15966A",
    "keep": "#234E8A",
    "bandit": "#397F8C",
    "fixed": "#A7AFBD",
    "edl": "#C84343",
    "positive": "#087B58",
    "negative": "#B53333",
}


def render(
    report_path: str | Path, output_path: str | Path | None = None
) -> dict[str, Any]:
    source = Path(report_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate_report(payload)
    test = payload["test_evaluation"]
    policies = test["policies"]

    output = (
        Path(output_path).resolve()
        if output_path is not None
        else source.with_name("route_policy_pi_summary.png")
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelcolor": COLORS["muted"],
            "xtick.color": COLORS["muted"],
            "ytick.color": COLORS["ink"],
        }
    )
    figure = plt.figure(figsize=(16, 9), facecolor=COLORS["background"])
    figure.add_artist(
        plt.Line2D([0.055, 0.945], [0.965, 0.965], color=COLORS["keep"], linewidth=4)
    )
    figure.text(
        0.055,
        0.925,
        "Frozen route test: oracle headroom exists — learned EDL routing failed",
        color=COLORS["ink"],
        fontsize=25,
        fontweight="bold",
        va="center",
    )
    figure.text(
        0.055,
        0.882,
        "ResEnc + AutoPET V composition  •  4 studies / 2 patients  •  patient-disjoint frozen test",
        color=COLORS["muted"],
        fontsize=12.5,
        va="center",
    )
    figure.text(
        0.94,
        0.875,
        "INTERNAL · PRIOR-EXPOSED",
        ha="right",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color="#8E2F2F",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#FBEAEA",
            "edgecolor": "#E7B9B9",
        },
    )

    _draw_mean_dice_panel(figure, policies)
    _draw_readout_panel(figure, policies)
    _draw_study_panel(figure, test["per_study"])

    caveat = (
        "CAVEAT  n=2 prior-exposed test patients (4 studies), below the prespecified minimum n=20. "
        "Internal exploratory evidence only — no efficacy, external-validation, learned-STOP, "
        "online-RL, clinical-generalization, or deployment claim."
    )
    figure.text(
        0.055,
        0.032,
        caveat,
        fontsize=9.6,
        color="#6F3030",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#FFF4F2",
            "edgecolor": "#E8C4BE",
        },
    )
    report_digest = _sha256(source)
    figure.text(
        0.945,
        0.008,
        f"Source: {source.name}  ·  SHA-256 {report_digest[:12]}…",
        fontsize=7.5,
        color="#8A93A3",
        ha="right",
    )
    figure.savefig(
        output,
        dpi=160,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.18,
        metadata={
            "Title": "Frozen route-policy PI summary",
            "Description": payload["claim_boundary"],
            "Source": str(source),
        },
    )
    plt.close(figure)

    summary = {
        "output": str(output),
        "bytes": output.stat().st_size,
        "sha256": _sha256(output),
        "report_sha256": report_digest,
        "plotted": {
            key: {
                "mean_final_dice": _study(policies[policy])["mean_final_dice"],
                "mean_delta_dice": _study(policies[policy])["mean_delta_dice"],
                "coverage": policies[policy]["coverage"],
                "harmful_action_rate": policies[policy][
                    "harmful_action_rate_all_studies"
                ],
            }
            for key, policy in POLICIES.items()
        },
        "test_studies": test["study_count"],
        "test_patients": test["patient_count"],
        "status": payload["status"],
    }
    print(json.dumps(summary, indent=2))
    return summary


def _draw_mean_dice_panel(figure: plt.Figure, policies: Mapping[str, Any]) -> None:
    axis = figure.add_axes([0.06, 0.46, 0.56, 0.35], facecolor=COLORS["panel"])
    names = [
        "Hindsight oracle (GT upper bound)",
        "KEEP ResEnc",
        "Linear bandit",
        "Best fixed: R2 intersection",
        "EDL router",
    ]
    keys = ["oracle", "keep", "bandit", "fixed", "edl"]
    values = [_study(policies[POLICIES[key]])["mean_final_dice"] for key in keys]
    deltas = [_study(policies[POLICIES[key]])["mean_delta_dice"] for key in keys]
    colors = [COLORS[key] for key in keys]
    y = np.arange(len(names))
    axis.barh(y, values, height=0.58, color=colors, edgecolor="none", zorder=3)
    baseline = _study(policies[POLICIES["keep"]])["mean_final_dice"]
    axis.axvline(
        baseline, color=COLORS["keep"], linewidth=1.6, linestyle=(0, (3, 3)), zorder=2
    )
    axis.text(
        baseline,
        -0.72,
        f"KEEP reference  {baseline:.3f}",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color=COLORS["keep"],
        fontweight="bold",
    )
    for row, (value, delta, key) in enumerate(zip(values, deltas, keys, strict=True)):
        delta_text = "baseline" if key == "keep" else f"Δ {delta:+.3f}"
        axis.text(
            min(value + 0.014, 0.727),
            row,
            f"{value:.3f}   {delta_text}",
            va="center",
            ha="left" if value < 0.71 else "right",
            fontsize=10.8,
            color=COLORS["ink"],
            fontweight="bold" if key in {"oracle", "edl"} else "normal",
        )
    axis.set_yticks(y, labels=names, fontsize=11.5)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 0.75)
    axis.set_xticks(np.arange(0.0, 0.76, 0.1))
    axis.set_xlabel("Mean Dice (absolute; zero-based axis)", fontsize=10)
    axis.set_title(
        "Frozen-test segmentation outcome", loc="left", pad=17, color=COLORS["ink"]
    )
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(COLORS["grid"])
    axis.tick_params(axis="y", length=0, pad=9)


def _draw_readout_panel(figure: plt.Figure, policies: Mapping[str, Any]) -> None:
    axis = figure.add_axes([0.655, 0.46, 0.29, 0.35])
    axis.set_axis_off()
    axis.text(
        0.0,
        1.03,
        "Decision readout",
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )

    oracle = policies[POLICIES["oracle"]]
    edl = policies[POLICIES["edl"]]
    bandit = policies[POLICIES["bandit"]]
    cards = [
        (
            0.68,
            "#E8F5EF",
            "#B9DDCF",
            COLORS["oracle"],
            f"{_study(oracle)['mean_delta_dice']:+.3f}",
            "Oracle headroom (GT-only)",
            "3 wins · 1 tie · 0 losses\n75% coverage · 0 harmful · nondeployable",
        ),
        (
            0.35,
            "#FBEDED",
            "#E7BBBB",
            COLORS["edl"],
            f"{_study(edl)['mean_delta_dice']:+.3f}",
            "EDL router failed",
            "2 wins · 2 losses\n100% coverage · 50% harmful",
        ),
        (
            0.02,
            "#EBF3F4",
            "#C4DADD",
            COLORS["bandit"],
            f"{_study(bandit)['mean_delta_dice']:+.3f}",
            "Bandit abstained safely",
            "0% coverage · 0 harmful\nNo gain — equivalent to KEEP",
        ),
    ]
    for y, face, edge, accent, number, title, detail in cards:
        axis.add_patch(
            FancyBboxPatch(
                (0.0, y),
                1.0,
                0.27,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                transform=axis.transAxes,
                facecolor=face,
                edgecolor=edge,
                linewidth=1.1,
            )
        )
        axis.text(
            0.055,
            y + 0.175,
            number,
            transform=axis.transAxes,
            fontsize=20,
            fontweight="bold",
            color=accent,
        )
        axis.text(
            0.31,
            y + 0.19,
            title,
            transform=axis.transAxes,
            fontsize=11.5,
            fontweight="bold",
            color=COLORS["ink"],
        )
        axis.text(
            0.31,
            y + 0.075,
            detail,
            transform=axis.transAxes,
            fontsize=9.5,
            color=COLORS["muted"],
            linespacing=1.35,
        )


def _draw_study_panel(
    figure: plt.Figure, per_study: Sequence[Mapping[str, Any]]
) -> None:
    axis = figure.add_axes([0.06, 0.105, 0.885, 0.27], facecolor=COLORS["panel"])
    axis.set_axis_off()
    axis.text(
        0.0,
        1.06,
        "Per-study selected route and ΔDice",
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    axis.text(
        1.0,
        1.06,
        "Bandit selected KEEP for all four studies",
        transform=axis.transAxes,
        fontsize=9.8,
        color=COLORS["bandit"],
        ha="right",
        fontweight="bold",
    )
    rows_by_policy: dict[str, dict[str, Mapping[str, Any]]] = {}
    for policy in (POLICIES["keep"], POLICIES["oracle"], POLICIES["edl"]):
        rows_by_policy[policy] = {
            str(row["case_id"]): row for row in per_study if row["policy"] == policy
        }
    case_ids = sorted(rows_by_policy[POLICIES["keep"]])
    cell_text = []
    for case_id in case_ids:
        keep = rows_by_policy[POLICIES["keep"]][case_id]
        oracle = rows_by_policy[POLICIES["oracle"]][case_id]
        edl = rows_by_policy[POLICIES["edl"]][case_id]
        cell_text.append(
            [
                case_id.replace("train_", ""),
                f"{float(keep['baseline_dice']):.3f}",
                _route_label(oracle),
                f"{float(oracle['delta_dice']):+.3f}",
                _route_label(edl),
                f"{float(edl['delta_dice']):+.3f}",
                "KEEP  +0.000",
            ]
        )
    table = axis.table(
        cellText=cell_text,
        colLabels=[
            "Study",
            "ResEnc Dice",
            "Oracle action (GT)",
            "Oracle Δ",
            "EDL action",
            "EDL Δ",
            "Bandit",
        ],
        colWidths=[0.12, 0.12, 0.17, 0.10, 0.17, 0.10, 0.16],
        cellLoc="center",
        loc="upper center",
        bbox=[0.0, 0.03, 1.0, 0.88],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS["grid"])
        cell.set_linewidth(0.7)
        if row == 0:
            cell.set_facecolor("#E9EDF3")
            cell.set_text_props(color=COLORS["ink"], fontweight="bold")
        else:
            cell.set_facecolor("#FFFFFF" if row % 2 else "#F7F9FB")
            if column == 3:
                value = float(cell_text[row - 1][column])
                cell.set_text_props(
                    color=COLORS["positive"] if value > 0 else COLORS["muted"],
                    fontweight="bold",
                )
            elif column == 5:
                value = float(cell_text[row - 1][column])
                cell.set_text_props(
                    color=COLORS["positive"] if value > 0 else COLORS["negative"],
                    fontweight="bold",
                )
            elif column == 6:
                cell.set_text_props(color=COLORS["bandit"], fontweight="bold")


def _route_label(row: Mapping[str, Any]) -> str:
    if row["selected_route"] == "KEEP":
        return "KEEP"
    action = str(row["selected_action"])
    short = {"intersection": "intersect", "replace": "replace", "union": "union"}[
        action
    ]
    return f"R{int(row['selected_round'])} {short}"


def _study(policy: Mapping[str, Any]) -> Mapping[str, Any]:
    return policy["study_estimand"]


def _validate_report(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "EXPLORATORY_INTERNAL_PRIOR_EXPOSED":
        raise ValueError(
            "renderer is scoped to the frozen prior-exposed internal report"
        )
    if bool(payload.get("efficacy_claim_eligible")) or bool(
        payload.get("external_validation_eligible")
    ):
        raise ValueError(
            "report claim boundary changed; refuse to render stale caveat language"
        )
    test = payload.get("test_evaluation", {})
    if test.get("study_count") != 4 or test.get("patient_count") != 2:
        raise ValueError("expected the frozen 4-study/2-patient test report")
    policies = test.get("policies", {})
    missing = sorted(set(POLICIES.values()) - set(policies))
    if missing:
        raise ValueError(f"report is missing required policies: {missing}")
    numeric = []
    for policy in POLICIES.values():
        numeric.extend(
            [
                _study(policies[policy])["mean_final_dice"],
                _study(policies[policy])["mean_delta_dice"],
                policies[policy]["coverage"],
                policies[policy]["harmful_action_rate_all_studies"],
            ]
        )
    if not all(math.isfinite(float(value)) for value in numeric):
        raise ValueError("report contains non-finite plotted values")
    if _study(policies[POLICIES["oracle"]])["mean_delta_dice"] <= 0:
        raise ValueError("report no longer shows positive oracle headroom")
    if _study(policies[POLICIES["edl"]])["mean_delta_dice"] >= 0:
        raise ValueError("report no longer supports the EDL failure heading")
    if not math.isclose(
        _study(policies[POLICIES["bandit"]])["mean_delta_dice"], 0.0, abs_tol=1e-12
    ) or not math.isclose(policies[POLICIES["bandit"]]["coverage"], 0.0, abs_tol=1e-12):
        raise ValueError("report no longer supports the safe-abstention bandit heading")
    expected_rows = 4 * len(policies)
    if len(test.get("per_study", [])) != expected_rows:
        raise ValueError("per-study row count does not match policy/study counts")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    render(args.report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
