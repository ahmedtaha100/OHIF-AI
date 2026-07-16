"""Render a sealed-safe, aggregate-only PI summary for fusion-only routing."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


FUSION_ROUTES = (
    "KEEP",
    "r1_intersection",
    "r2_intersection",
    "r1_union",
    "r2_union",
)
LEARNED_POLICIES = (
    "edl_accept_best_utility",
    "linear_contextual_bandit",
)

COLORS = {
    "ink": "#172033",
    "muted": "#677185",
    "grid": "#DDE2EA",
    "background": "#F7F8FA",
    "panel": "#FFFFFF",
    "keep": "#234E8A",
    "edl": "#16858F",
    "ridge": "#7356A8",
    "fixed": "#A7AFBD",
    "oracle": "#15966A",
    "positive": "#087B58",
    "warning": "#A35B12",
    "negative": "#B53333",
}


def render(
    report_path: str | Path,
    output_path: str | Path | None = None,
    markdown_path: str | Path | None = None,
) -> dict[str, Any]:
    """Render aggregate metrics without dereferencing any path inside the report."""

    source = Path(report_path).resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    _validate_report(payload)
    view = _aggregate_view(payload)

    output = (
        Path(output_path).resolve()
        if output_path is not None
        else source.with_name("fusion_only_pi_summary.png")
    )
    markdown = (
        Path(markdown_path).resolve()
        if markdown_path is not None
        else source.with_name("fusion_only_pi_summary.md")
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown.parent.mkdir(parents=True, exist_ok=True)

    report_digest = _sha256(source)
    figure = _build_figure(view, source.name, report_digest)
    temporary_output = output.with_name(f"{output.stem}.tmp{output.suffix}")
    try:
        figure.savefig(
            temporary_output,
            dpi=160,
            facecolor=figure.get_facecolor(),
            bbox_inches="tight",
            pad_inches=0.18,
            metadata={
                "Title": "Fusion-only route-policy PI summary",
                "Description": view["claim_boundary"],
                "Source": source.name,
            },
        )
        os.replace(temporary_output, output)
    finally:
        plt.close(figure)
        temporary_output.unlink(missing_ok=True)

    markdown_text = _markdown(view, source.name, report_digest, output.name)
    temporary_markdown = markdown.with_name(f"{markdown.name}.tmp")
    try:
        temporary_markdown.write_text(markdown_text, encoding="utf-8")
        os.replace(temporary_markdown, markdown)
    finally:
        temporary_markdown.unlink(missing_ok=True)

    summary = {
        "status": "SYNTHETIC_RENDER_VALIDATED"
        if view["synthetic"]
        else "PI_SUMMARY_RENDERED",
        "output": str(output),
        "output_sha256": _sha256(output),
        "markdown": str(markdown),
        "markdown_sha256": _sha256(markdown),
        "report_sha256": report_digest,
        "synthetic": view["synthetic"],
        "verdict": view["verdict"],
        "test_patients": view["patient_count"],
        "test_studies": view["study_count"],
        "edl_delta_dice": view["policies"]["edl"]["mean_delta_dice"],
        "ridge_delta_dice": view["policies"]["ridge"]["mean_delta_dice"],
    }
    print(json.dumps(summary, indent=2))
    return summary


def _aggregate_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Copy only cohort-level fields used by the PI artifact."""

    test = payload["test_evaluation"]
    raw_policies = test["policies"]
    fixed_key = max(
        (f"fixed_{route}" for route in FUSION_ROUTES if route != "KEEP"),
        key=lambda key: float(raw_policies[key]["patient_estimand"]["mean_final_dice"]),
    )
    selected = {
        "keep": "keep_resenc",
        "edl": "edl_accept_best_utility",
        "ridge": "linear_contextual_bandit",
        "fixed": fixed_key,
        "oracle": "hindsight_oracle",
    }
    policies = {
        label: _policy_projection(raw_policies[key]) for label, key in selected.items()
    }
    edl_deploy_keep_all = bool(payload["edl"]["deploy_keep_all"])
    verdict, tone = _verdict(policies["edl"], edl_deploy_keep_all)
    if bool(payload.get("synthetic", False)):
        verdict = f"Synthetic renderer check - {verdict.lower()}"
    return {
        "synthetic": bool(payload.get("synthetic", False)),
        "status": str(payload["status"]),
        "claim_boundary": str(payload["claim_boundary"]),
        "patient_count": int(test["patient_count"]),
        "study_count": int(test["study_count"]),
        "candidate_routes": tuple(test["candidate_routes"]),
        "test_label_evaluation_passes": int(test["test_label_evaluation_passes"]),
        "deployment_frozen_before_test": bool(
            payload["deployment_artifact"]["frozen_before_test_manifest_open"]
        ),
        "test_used_for_selection": bool(
            payload["no_test_tuning_audit"][
                "test_used_for_model_or_threshold_selection"
            ]
        ),
        "edl_development_decision": str(
            payload["edl"]["selection"]["deployment_decision"]
        ),
        "edl_deploy_keep_all": edl_deploy_keep_all,
        "ridge_development_decision": str(
            payload["linear_contextual_bandit"]["deployment_decision"]
        ),
        "fixed_policy": fixed_key,
        "policies": policies,
        "verdict": verdict,
        "verdict_tone": tone,
    }


def _policy_projection(policy: Mapping[str, Any]) -> dict[str, Any]:
    patient = policy["patient_estimand"]
    study = policy["study_estimand"]
    ci = patient["paired_bootstrap_95_ci_delta_dice"]
    return {
        "mean_final_dice": float(patient["mean_final_dice"]),
        "mean_delta_dice": float(patient["mean_delta_dice"]),
        "mean_delta_nsd_2mm": float(patient["mean_delta_nsd_2mm"]),
        "ci_lower": float(ci["lower"]),
        "ci_upper": float(ci["upper"]),
        "wins": int(patient["dice_win_tie_loss_vs_keep"]["wins"]),
        "ties": int(patient["dice_win_tie_loss_vs_keep"]["ties"]),
        "losses": int(patient["dice_win_tie_loss_vs_keep"]["losses"]),
        "study_mean_delta_dice": float(study["mean_delta_dice"]),
        "coverage": float(policy["coverage"]),
        "harm_rate": float(policy["harmful_action_rate_all_studies"]),
    }


def _verdict(policy: Mapping[str, Any], deploy_keep_all: bool) -> tuple[str, str]:
    delta = float(policy["mean_delta_dice"])
    lower = float(policy["ci_lower"])
    harm = float(policy["harm_rate"])
    if deploy_keep_all:
        return "EDL safety gate abstained; no segmentation gain over KEEP", "warning"
    if delta > 0.0 and lower > 0.0 and harm <= 0.05:
        return (
            "EDL fusion routing improved segmentation on the internal test",
            "positive",
        )
    if delta > 0.0:
        return (
            "EDL point estimate improved, but uncertainty includes no gain",
            "warning",
        )
    return "EDL fusion routing did not improve segmentation", "negative"


def _build_figure(
    view: Mapping[str, Any], source_name: str, report_digest: str
) -> plt.Figure:
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
        0.922,
        str(view["verdict"]),
        color=COLORS["ink"],
        fontsize=23,
        fontweight="bold",
        va="center",
    )
    figure.text(
        0.055,
        0.878,
        (
            "ResEnc + fusion-only prompt routes  |  "
            f"{view['study_count']} studies / {view['patient_count']} patients  |  "
            "patient-disjoint one-pass test"
        ),
        color=COLORS["muted"],
        fontsize=12.5,
        va="center",
    )
    badge = "SYNTHETIC SMOKE TEST" if view["synthetic"] else "INTERNAL - PRIOR-EXPOSED"
    badge_face = "#FFF4D6" if view["synthetic"] else "#FBEAEA"
    badge_edge = "#E5C879" if view["synthetic"] else "#E7B9B9"
    figure.text(
        0.94,
        0.875,
        badge,
        ha="right",
        va="center",
        fontsize=10.5,
        fontweight="bold",
        color=COLORS["warning"] if view["synthetic"] else "#8E2F2F",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": badge_face,
            "edgecolor": badge_edge,
        },
    )
    _draw_dice_panel(figure, view)
    _draw_readout_panel(figure, view)
    _draw_protocol_panel(figure, view)
    caveat = (
        "CAVEAT  Internal prior-exposed test; robot prompts are ground-truth-derived. "
        "Offline oracle-assisted evidence only - no external-validation, online-RL, "
        "clinical-efficacy, or deployment claim."
    )
    figure.text(
        0.055,
        0.038,
        caveat,
        fontsize=9.5,
        color="#6F3030",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.55",
            "facecolor": "#FFF4F2",
            "edgecolor": "#E8C4BE",
        },
    )
    figure.text(
        0.945,
        0.009,
        f"Source: {source_name}  |  SHA-256 {report_digest[:12]}...",
        fontsize=7.5,
        color="#8A93A3",
        ha="right",
    )
    return figure


def _draw_dice_panel(figure: plt.Figure, view: Mapping[str, Any]) -> None:
    policies = view["policies"]
    fixed_label = _fixed_label(str(view["fixed_policy"]))
    rows = (
        ("KEEP ResEnc", "keep"),
        ("EDL fusion gate", "edl"),
        ("Linear ridge comparator", "ridge"),
        (f"Best fixed: {fixed_label}", "fixed"),
        ("Hindsight oracle (GT upper bound)", "oracle"),
    )
    axis = figure.add_axes([0.06, 0.44, 0.56, 0.36], facecolor=COLORS["panel"])
    values = [policies[key]["mean_final_dice"] for _, key in rows]
    deltas = [policies[key]["mean_delta_dice"] for _, key in rows]
    colors = [COLORS[key] for _, key in rows]
    y = np.arange(len(rows))
    axis.barh(y, values, height=0.58, color=colors, edgecolor="none", zorder=3)
    baseline = policies["keep"]["mean_final_dice"]
    axis.axvline(baseline, color=COLORS["keep"], linewidth=1.5, linestyle=(0, (3, 3)))
    for row, (value, delta) in enumerate(zip(values, deltas, strict=True)):
        label = "baseline" if row == 0 else f"delta {delta:+.3f}"
        axis.text(
            min(value + 0.016, 0.965),
            row,
            f"{value:.3f}   {label}",
            va="center",
            ha="left" if value < 0.94 else "right",
            fontsize=10.5,
            color=COLORS["ink"],
            fontweight="bold" if row in {1, 4} else "normal",
        )
    axis.set_yticks(y, labels=[label for label, _ in rows], fontsize=11.2)
    axis.invert_yaxis()
    axis.set_xlim(0.0, 1.0)
    axis.set_xticks(np.arange(0.0, 1.01, 0.1))
    axis.set_xlabel("Patient-level mean Dice (absolute; zero-based axis)", fontsize=10)
    axis.set_title("One-pass internal test outcome", loc="left", pad=17)
    axis.grid(axis="x", color=COLORS["grid"], linewidth=0.8, zorder=0)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(COLORS["grid"])
    axis.tick_params(axis="y", length=0, pad=9)


def _draw_readout_panel(figure: plt.Figure, view: Mapping[str, Any]) -> None:
    axis = figure.add_axes([0.655, 0.44, 0.29, 0.36])
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
    cards = (
        (0.68, "edl", "EDL fusion gate", str(view["edl_development_decision"])),
        (
            0.35,
            "ridge",
            "Linear ridge comparator",
            f"{view['ridge_development_decision']} - not online RL",
        ),
        (0.02, "oracle", "Hindsight oracle", "GT-only upper bound - nondeployable"),
    )
    for y, key, title, decision in cards:
        policy = view["policies"][key]
        accent = COLORS[key]
        axis.add_patch(
            FancyBboxPatch(
                (0.0, y),
                1.0,
                0.27,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                transform=axis.transAxes,
                facecolor="#FFFFFF",
                edgecolor=COLORS["grid"],
                linewidth=1.1,
            )
        )
        axis.text(
            0.045,
            y + 0.17,
            f"{policy['mean_delta_dice']:+.3f}",
            transform=axis.transAxes,
            fontsize=19,
            fontweight="bold",
            color=accent,
        )
        axis.text(
            0.30,
            y + 0.19,
            title,
            transform=axis.transAxes,
            fontsize=11.2,
            fontweight="bold",
            color=COLORS["ink"],
        )
        detail = (
            f"95% CI [{policy['ci_lower']:+.3f}, {policy['ci_upper']:+.3f}]  |  "
            f"harm {policy['harm_rate']:.1%}\n"
            f"coverage {policy['coverage']:.1%}  |  patient W/T/L "
            f"{policy['wins']}/{policy['ties']}/{policy['losses']}\n{decision}"
        )
        axis.text(
            0.30,
            y + 0.055,
            detail,
            transform=axis.transAxes,
            fontsize=8.7,
            color=COLORS["muted"],
            linespacing=1.28,
        )


def _draw_protocol_panel(figure: plt.Figure, view: Mapping[str, Any]) -> None:
    axis = figure.add_axes([0.06, 0.115, 0.885, 0.23])
    axis.set_axis_off()
    axis.text(
        0.0,
        1.04,
        "What this result can establish",
        transform=axis.transAxes,
        fontsize=15,
        fontweight="bold",
        color=COLORS["ink"],
    )
    items = (
        (
            "Frozen before test",
            "Yes" if view["deployment_frozen_before_test"] else "No",
            "EDL thresholds and ridge model were fixed in development.",
        ),
        (
            "Test access",
            f"{view['test_label_evaluation_passes']} pass",
            "No test outcome was used for model or threshold selection.",
        ),
        (
            "Action scope",
            "Fusion only",
            "KEEP plus R1/R2 intersection or union; replacement excluded.",
        ),
        (
            "Evidence boundary",
            "Internal",
            "Prior-exposed cases and GT-derived robot corrections.",
        ),
    )
    width = 0.235
    for index, (title, value, detail) in enumerate(items):
        x = index * 0.255
        axis.add_patch(
            FancyBboxPatch(
                (x, 0.05),
                width,
                0.78,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                transform=axis.transAxes,
                facecolor=COLORS["panel"],
                edgecolor=COLORS["grid"],
                linewidth=1.0,
            )
        )
        axis.text(
            x + 0.025,
            0.65,
            title,
            transform=axis.transAxes,
            fontsize=9.6,
            fontweight="bold",
            color=COLORS["muted"],
        )
        axis.text(
            x + 0.025,
            0.43,
            value,
            transform=axis.transAxes,
            fontsize=16,
            fontweight="bold",
            color=COLORS["keep"],
        )
        axis.text(
            x + 0.025,
            0.12,
            detail,
            transform=axis.transAxes,
            fontsize=8.8,
            color=COLORS["muted"],
            wrap=True,
        )


def _markdown(
    view: Mapping[str, Any],
    source_name: str,
    report_digest: str,
    image_name: str,
) -> str:
    policies = view["policies"]
    prefix = (
        "> [!CAUTION]\n> SYNTHETIC SMOKE TEST - NOT A RESEARCH RESULT.\n\n"
        if view["synthetic"]
        else ""
    )
    rows = []
    for label, key in (
        ("KEEP ResEnc", "keep"),
        ("EDL fusion gate", "edl"),
        ("Linear ridge comparator (not online RL)", "ridge"),
        (f"Best fixed: {_fixed_label(view['fixed_policy'])}", "fixed"),
        ("Hindsight oracle (GT upper bound)", "oracle"),
    ):
        policy = policies[key]
        rows.append(
            f"| {label} | {policy['mean_final_dice']:.4f} | "
            f"{policy['mean_delta_dice']:+.4f} | "
            f"[{policy['ci_lower']:+.4f}, {policy['ci_upper']:+.4f}] | "
            f"{policy['coverage']:.1%} | {policy['harm_rate']:.1%} | "
            f"{policy['wins']}/{policy['ties']}/{policy['losses']} |"
        )
    return (
        "# Fusion-only segmentation routing - PI readout\n\n"
        f"{prefix}"
        f"![Fusion-only PI summary]({image_name})\n\n"
        f"**Verdict:** {view['verdict']}\n\n"
        f"Frozen internal test: {view['patient_count']} patients / "
        f"{view['study_count']} studies.\n\n"
        "| Policy | Patient mean Dice | Delta Dice | Patient-bootstrap 95% CI | "
        "Coverage | Harm rate | Patient W/T/L |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n" + "\n".join(rows) + "\n\n"
        "## Interpretation\n\n"
        f"- EDL development decision: `{view['edl_development_decision']}`.\n"
        f"- Linear ridge development decision: `{view['ridge_development_decision']}`. "
        "This is a full-information offline comparator, not online RL.\n"
        "- Hindsight oracle uses realized ground-truth utility and is an upper bound, "
        "not a deployable policy.\n\n"
        "## Protocol integrity\n\n"
        f"- Deployment frozen before test manifest opening: "
        f"`{str(view['deployment_frozen_before_test']).lower()}`.\n"
        f"- Test label evaluation passes: `{view['test_label_evaluation_passes']}`.\n"
        f"- Test used for model/threshold selection: "
        f"`{str(view['test_used_for_selection']).lower()}`.\n"
        "- Action menu: KEEP plus R1/R2 intersection or union; replacement excluded.\n\n"
        "## Claim boundary\n\n"
        f"{view['claim_boundary']} Robot corrections are ground-truth-derived, so this "
        "remains offline oracle-assisted evidence. No external-validation, online-RL, "
        "clinical-efficacy, or deployment claim.\n\n"
        f"Source: `{source_name}`  \n"
        f"Source SHA-256: `{report_digest}`\n"
    )


def _validate_report(payload: Mapping[str, Any]) -> None:
    if payload.get("status") != "EXPLORATORY_INTERNAL_PRIOR_EXPOSED":
        raise ValueError("renderer is scoped to the prior-exposed fusion-only rescue")
    if bool(payload.get("efficacy_claim_eligible")) or bool(
        payload.get("external_validation_eligible")
    ):
        raise ValueError(
            "refuse to render internal data with an efficacy/external claim"
        )
    test = _require_mapping(payload.get("test_evaluation"), "test_evaluation")
    if test.get("scope") != "frozen_test":
        raise ValueError("expected frozen_test scope")
    if (
        int(test.get("patient_count", -1)) != 6
        or int(test.get("study_count", -1)) != 12
    ):
        raise ValueError("expected the frozen fusion-only 6-patient/12-study test")
    if int(test.get("test_label_evaluation_passes", -1)) != 1:
        raise ValueError("expected exactly one test-label evaluation pass")
    candidate_routes = tuple(str(route) for route in test.get("candidate_routes", ()))
    if candidate_routes != FUSION_ROUTES:
        raise ValueError(f"expected exact fusion-only route menu {list(FUSION_ROUTES)}")
    if any("replace" in route.lower() for route in candidate_routes):
        raise ValueError("replacement actions are forbidden in fusion-only reporting")
    deployment = _require_mapping(
        payload.get("deployment_artifact"), "deployment_artifact"
    )
    if deployment.get("frozen_before_test_manifest_open") is not True:
        raise ValueError("deployment was not frozen before test manifest opening")
    audit = _require_mapping(
        payload.get("no_test_tuning_audit"), "no_test_tuning_audit"
    )
    if int(audit.get("test_label_evaluation_passes", -1)) != 1:
        raise ValueError("no-test-tuning audit does not confirm exactly one test pass")
    if audit.get("test_used_for_model_or_threshold_selection") is not False:
        raise ValueError("test outcomes entered model or threshold selection")
    edl = _require_mapping(payload.get("edl"), "edl")
    selection = _require_mapping(edl.get("selection"), "edl.selection")
    if not isinstance(edl.get("deploy_keep_all"), bool):
        raise ValueError("edl.deploy_keep_all must be boolean")
    if not isinstance(selection.get("safety_deployed"), bool):
        raise ValueError("edl.selection.safety_deployed must be boolean")
    if bool(edl["deploy_keep_all"]) == bool(selection["safety_deployed"]):
        raise ValueError("EDL deploy/fallback flags are inconsistent")
    if not str(selection.get("deployment_decision", "")):
        raise ValueError("missing EDL development decision")
    ridge = _require_mapping(
        payload.get("linear_contextual_bandit"), "linear_contextual_bandit"
    )
    if not str(ridge.get("deployment_decision", "")):
        raise ValueError("missing ridge development decision")
    policies = _require_mapping(test.get("policies"), "test_evaluation.policies")
    required = {
        "keep_resenc",
        "hindsight_oracle",
        *LEARNED_POLICIES,
        *(f"fixed_{route}" for route in FUSION_ROUTES if route != "KEEP"),
    }
    missing = sorted(required - set(policies))
    if missing:
        raise ValueError(f"report is missing policies: {missing}")
    for name in sorted(required):
        _validate_policy(name, _require_mapping(policies[name], f"policies.{name}"))
    keep = policies["keep_resenc"]
    if not math.isclose(
        float(keep["patient_estimand"]["mean_delta_dice"]), 0.0, abs_tol=1e-12
    ):
        raise ValueError("KEEP must have zero delta Dice")
    if not math.isclose(float(keep["coverage"]), 0.0, abs_tol=1e-12):
        raise ValueError("KEEP must have zero coverage")
    if not math.isclose(
        float(keep["harmful_action_rate_all_studies"]), 0.0, abs_tol=1e-12
    ):
        raise ValueError("KEEP must have zero harm")


def _validate_policy(name: str, policy: Mapping[str, Any]) -> None:
    patient = _require_mapping(
        policy.get("patient_estimand"), f"{name}.patient_estimand"
    )
    study = _require_mapping(policy.get("study_estimand"), f"{name}.study_estimand")
    if int(patient.get("n", -1)) != 6 or int(study.get("n", -1)) != 12:
        raise ValueError(f"{name} estimand counts do not match the frozen cohort")
    ci = _require_mapping(
        patient.get("paired_bootstrap_95_ci_delta_dice"), f"{name}.dice_ci"
    )
    wtl = _require_mapping(
        patient.get("dice_win_tie_loss_vs_keep"), f"{name}.patient_wtl"
    )
    numeric = {
        "patient mean Dice": patient.get("mean_final_dice"),
        "patient delta Dice": patient.get("mean_delta_dice"),
        "patient delta NSD": patient.get("mean_delta_nsd_2mm"),
        "study delta Dice": study.get("mean_delta_dice"),
        "CI lower": ci.get("lower"),
        "CI upper": ci.get("upper"),
        "coverage": policy.get("coverage"),
        "harm rate": policy.get("harmful_action_rate_all_studies"),
    }
    for field, value in numeric.items():
        if not _is_finite_number(value):
            raise ValueError(f"{name} has invalid {field}")
    if not 0.0 <= float(patient["mean_final_dice"]) <= 1.0:
        raise ValueError(f"{name} Dice is outside [0, 1]")
    for field in ("coverage", "harmful_action_rate_all_studies"):
        if not 0.0 <= float(policy[field]) <= 1.0:
            raise ValueError(f"{name} {field} is outside [0, 1]")
    if float(ci["lower"]) > float(ci["upper"]):
        raise ValueError(f"{name} has reversed confidence interval")
    counts = [int(wtl.get(key, -1)) for key in ("wins", "ties", "losses")]
    if min(counts) < 0 or sum(counts) != 6:
        raise ValueError(f"{name} patient win/tie/loss does not sum to 6")


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _fixed_label(policy: str) -> str:
    return (
        policy.removeprefix("fixed_")
        .replace("r1", "R1")
        .replace("r2", "R2")
        .replace("_", " ")
    )


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
    parser.add_argument("--markdown-output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    render(args.report, args.output, args.markdown_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
