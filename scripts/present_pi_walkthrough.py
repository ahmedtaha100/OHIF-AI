#!/usr/bin/env python3
"""Present the sealed segmentation and paper-reproduction findings in a terminal.

This is a dependency-free, screen-recording aid.  Its constants are copied from
the sealed 2026-07-15 experiment report and the completed public-artifact audit;
it does not recompute, reinterpret, or fetch research results at presentation time.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import shutil
import sys
import textwrap
import time
from typing import TextIO


ESC = "\x1b["


@dataclass(frozen=True)
class Cell:
    """One table cell with an optional semantic color."""

    text: str
    tone: str = "plain"


@dataclass(frozen=True)
class Theme:
    """Small ANSI theme that degrades to plain text deterministically."""

    enabled: bool

    _CODES = {
        "plain": "",
        "title": "1;96",
        "heading": "1;97",
        "muted": "2;37",
        "info": "36",
        "good": "1;32",
        "warn": "1;33",
        "bad": "1;31",
        "claim": "1;35",
        "evidence": "1;36",
        "limit": "1;33",
    }

    def paint(self, text: str, tone: str = "plain") -> str:
        code = self._CODES.get(tone, "")
        if not self.enabled or not code:
            return text
        return f"{ESC}{code}m{text}{ESC}0m"


@dataclass(frozen=True)
class Section:
    """A named presentation section."""

    key: str
    title: str
    summary: str
    renderer: Callable[[int, Theme], list[str]]


def _cell(value: str | Cell) -> Cell:
    return value if isinstance(value, Cell) else Cell(str(value))


def _wrap(text: str, width: int) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        lines.extend(
            textwrap.wrap(
                paragraph,
                width=max(1, width),
                break_long_words=True,
                break_on_hyphens=False,
                replace_whitespace=False,
            )
            or [""]
        )
    return lines


def _rule(width: int, character: str = "-") -> str:
    return character * width


def _banner(title: str, subtitle: str, width: int, theme: Theme) -> list[str]:
    inner = width - 4
    title_lines = _wrap(title, inner)
    subtitle_lines = _wrap(subtitle, inner)
    result = [theme.paint("+" + _rule(width - 2, "=") + "+", "title")]
    for line in title_lines:
        padding = inner - len(line)
        result.append(
            theme.paint("| ", "title")
            + theme.paint(line + " " * padding, "title")
            + theme.paint(" |", "title")
        )
    result.append(theme.paint("| " + _rule(inner) + " |", "title"))
    for line in subtitle_lines:
        padding = inner - len(line)
        result.append(
            theme.paint("| ", "title")
            + theme.paint(line + " " * padding, "muted")
            + theme.paint(" |", "title")
        )
    result.append(theme.paint("+" + _rule(width - 2, "=") + "+", "title"))
    return result


def _callout(label: str, text: str, width: int, theme: Theme, tone: str) -> list[str]:
    prefix = f"[{label}] "
    wrapped = _wrap(text, width - len(prefix) - 2)
    lines: list[str] = []
    for index, line in enumerate(wrapped):
        current_prefix = prefix if index == 0 else " " * len(prefix)
        lines.append(
            theme.paint(current_prefix, tone)
            + theme.paint(line, tone if tone in {"good", "warn", "bad"} else "plain")
        )
    return lines


def _bullets(
    items: Sequence[str | Cell], width: int, theme: Theme, marker: str = "-"
) -> list[str]:
    result: list[str] = []
    for item_value in items:
        item = _cell(item_value)
        wrapped = _wrap(item.text, width - 4)
        for index, line in enumerate(wrapped):
            prefix = f"{marker} " if index == 0 else "  "
            result.append(prefix + theme.paint(line, item.tone))
    return result


def _table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str | Cell]],
    widths: Sequence[int],
    theme: Theme,
) -> list[str]:
    if len(headers) != len(widths):
        raise ValueError("table headers and widths differ")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("table row width differs from headers")

    separator = "+" + "+".join(_rule(width + 2) for width in widths) + "+"
    result = [theme.paint(separator, "muted")]
    wrapped_headers = [
        _wrap(header, width) for header, width in zip(headers, widths, strict=True)
    ]
    header_height = max(len(lines) for lines in wrapped_headers)
    for line_index in range(header_height):
        header_cells = []
        for lines, width in zip(wrapped_headers, widths, strict=True):
            value = lines[line_index] if line_index < len(lines) else ""
            header_cells.append(" " + theme.paint(value.ljust(width), "heading") + " ")
        result.append(
            theme.paint("|", "muted")
            + theme.paint("|", "muted").join(header_cells)
            + theme.paint("|", "muted")
        )
    result.append(theme.paint(separator, "muted"))

    for raw_row in rows:
        cells = [_cell(value) for value in raw_row]
        wrapped = [_wrap(cell.text, width) for cell, width in zip(cells, widths, strict=True)]
        row_height = max(len(lines) for lines in wrapped)
        for line_index in range(row_height):
            rendered = []
            for cell, lines, width in zip(cells, wrapped, widths, strict=True):
                value = lines[line_index] if line_index < len(lines) else ""
                rendered.append(" " + theme.paint(value.ljust(width), cell.tone) + " ")
            result.append(theme.paint("|", "muted") + theme.paint("|", "muted").join(rendered) + theme.paint("|", "muted"))
        result.append(theme.paint(separator, "muted"))
    return result


def _fit_widths(width: int, ratios: Sequence[float], minimums: Sequence[int]) -> list[int]:
    """Fit table content widths inside the requested terminal width."""

    if len(ratios) != len(minimums):
        raise ValueError("ratios and minimums differ")
    available = width - (3 * len(ratios) + 1)
    widths = [max(minimum, int(available * ratio)) for ratio, minimum in zip(ratios, minimums, strict=True)]
    while sum(widths) > available:
        candidate = max(
            (index for index, value in enumerate(widths) if value > minimums[index]),
            key=lambda index: widths[index] - minimums[index],
            default=None,
        )
        if candidate is None:
            break
        widths[candidate] -= 1
    while sum(widths) < available:
        candidate = max(range(len(widths)), key=lambda index: ratios[index])
        widths[candidate] += 1
    return widths


def _opening(width: int, theme: Theme) -> list[str]:
    lines = _banner(
        "SEGMENTATION POLICY + PAPER REPRODUCTION AUDIT",
        "PI walkthrough | sealed internal experiment and public-artifact audit | 2026-07-15",
        width,
        theme,
    )
    lines.extend(
        [
            "",
            theme.paint("EVIDENCE LABELS", "heading"),
            theme.paint("[MEASURED]", "evidence") + " sealed experiment output",
            theme.paint("[RECOMPUTED]", "evidence") + " independently derived from released artifacts",
            theme.paint("[MANUSCRIPT]", "claim") + " statement reported by the paper",
            theme.paint("[RELEASE]", "info") + " directly observed in the public repository",
            theme.paint("[LIMIT]", "limit") + " boundary on what the evidence can establish",
            "",
        ]
    )
    return lines


def _pipeline(width: int, theme: Theme) -> list[str]:
    lines = _banner(
        "1. FROM BASELINE TO A SEALED DECISION",
        "We evaluated increasingly constrained ways to improve segmentation, then audited the source paper.",
        width,
        theme,
    )
    widths = _fit_widths(width, (0.15, 0.50, 0.35), (10, 28, 20))
    lines.extend(
        _table(
            ("Stage", "What we did", "Decision gate"),
            (
                (
                    Cell("Baseline", "heading"),
                    "Ran the ResEnc-L segmentation and kept its prediction as the safe reference.",
                    Cell("KEEP is the comparator", "info"),
                ),
                (
                    Cell("Routes", "heading"),
                    "Constructed prompt refinements, intersections, unions, and replacement candidates; unsafe replacement was later excluded.",
                    Cell("Can any route beat KEEP?", "info"),
                ),
                (
                    Cell("EDL", "heading"),
                    "Estimated route confidence and used abstention/safety gates so uncertainty could return KEEP.",
                    Cell("Improve Dice without harm", "info"),
                ),
                (
                    Cell("RL-style", "heading"),
                    "Compared rule-based and learned offline route selectors, including a contextual-bandit comparator and oracle upper bounds.",
                    Cell("No test tuning", "info"),
                ),
                (
                    Cell("Sealed test", "heading"),
                    "Froze the hybrid policy before opening the 6-patient / 12-study test and ran one canonical test-label pass.",
                    Cell("Primary Dice + harm gate", "info"),
                ),
                (
                    Cell("Paper audit", "heading"),
                    "Inspected the linked implementation, weights, evaluator, released predictions, claims, and statistics.",
                    Cell("Reproduce or verify claims", "info"),
                ),
            ),
            widths,
            theme,
        )
    )
    lines.extend(["", *_callout(
        "LIMIT",
        "The segmentation cohort was internal and prior-exposed, and robot prompts were ground-truth-derived. This supports an offline policy decision, not a clinical-efficacy or external-validation claim.",
        width,
        theme,
        "limit",
    )])
    return lines


def _policy_result(width: int, theme: Theme) -> list[str]:
    lines = _banner(
        "2. SEALED RL / EDL POLICY RESULT",
        "The primary endpoint was Dice. The policy also had to satisfy a zero-harm gate.",
        width,
        theme,
    )
    widths = _fit_widths(width, (0.25, 0.22, 0.25, 0.28), (17, 14, 17, 19))
    lines.extend(
        _table(
            ("Metric", "KEEP baseline", "Primary hybrid", "Secondary screen"),
            (
                ("Mean Dice", "0.608227", Cell("0.605023", "bad"), Cell("0.604677*", "bad")),
                ("Delta Dice", "0.000000", Cell("-0.003204", "bad"), Cell("-0.003550", "bad")),
                ("Patient bootstrap 95% CI", "[0, 0]", Cell("[-0.009612, 0]", "bad"), Cell("[-0.009958, -0.000129]", "bad")),
                ("Action coverage", "0 / 12", "1 / 12", "3 / 12"),
                ("Harmful studies", "0 / 12", Cell("1 / 12", "bad"), Cell("3 / 12", "bad")),
                ("Patient W / T / L", "0 / 6 / 0", "0 / 5 / 1", Cell("0 / 3 / 3", "bad")),
                ("Decision", "Reference", Cell("FAIL", "bad"), Cell("FAIL", "bad")),
            ),
            widths,
            theme,
        )
    )
    lines.extend(
        [
            "",
            *_callout(
                "MEASURED",
                "The primary policy changed one study and harmed that study. The broader screen changed three studies and all three changes were harmful.",
                width,
                theme,
                "bad",
            ),
            *_callout(
                "MEASURED",
                "Primary NSD improved by +0.003509, but that secondary surface metric cannot rescue a failed primary Dice endpoint and failed harm gate.",
                width,
                theme,
                "warn",
            ),
            *_callout(
                "DECISION",
                "Keep ResEnc-L / KEEP. Do not claim an RL or EDL segmentation benefit and do not train PPO from this failed route-selection formulation.",
                width,
                theme,
                "bad",
            ),
            theme.paint("* Secondary final Dice is baseline plus the audited delta, shown for visual comparison.", "muted"),
        ]
    )
    return lines


def _paper_claims(width: int, theme: Theme) -> list[str]:
    lines = _banner(
        "3. PAPER CLAIMS VERSUS THE PUBLIC RELEASE",
        "Question: can the linked materials independently reproduce or verify the headline results?",
        width,
        theme,
    )
    lines.extend(
        [
            *_callout(
                "PAPER",
                "Promptable segmentation with region exploration enables minimal-effort expert-level prostate cancer delineation | DOI 10.1007/s11548-026-03628-w",
                width,
                theme,
                "claim",
            ),
            "",
        ]
    )
    widths = _fit_widths(width, (0.31, 0.30, 0.39), (20, 19, 25))
    lines.extend(
        _table(
            ("Paper claim / requirement", "Reported or expected", "What the public release supports"),
            (
                ("PROMIS performance", Cell("Dice 0.526", "claim"), Cell("Unverifiable", "bad")),
                ("PI-CAI performance", Cell("Dice 0.566", "claim"), Cell("Unverifiable", "bad")),
                ("Gain over Swin-UNETR", Cell("+0.099 / +0.089", "claim"), "Arithmetic is correct; experiment is not reproducible"),
                ("Exact holdout cohorts", "114 PROMIS; 218 PI-CAI implied", Cell("No paper cohort/split manifests", "bad")),
                ("Final trained policy", "Paper-matching PPO checkpoint", Cell("Not released", "bad")),
                ("Evaluation pathway", "Paper-matching evaluator", Cell("Hard-coded private paths; unusable as released", "bad")),
                ("Headline per-case outputs", "Predictions, metrics, selection rule", Cell("Not released", "bad")),
            ),
            widths,
            theme,
        )
    )
    lines.extend(
        [
            "",
            *_callout(
                "RECOMPUTED",
                "Seven bundled prediction/ground-truth pairs are selected training snapshots from only six unique cases. Released-code Dice mean = 0.314680; range = 0.233925 to 0.385494.",
                width,
                theme,
                "evidence",
            ),
            *_callout(
                "LIMIT",
                "Those seven files are not the claimed holdout cohort, so their lower Dice does not by itself falsify 0.526 or 0.566. It shows that the release contains no artifact that verifies either headline.",
                width,
                theme,
                "limit",
            ),
        ]
    )
    return lines


def _release_forensics(width: int, theme: Theme) -> list[str]:
    lines = _banner(
        "4. WHY THE RELEASE CANNOT RUN THE CLAIMED EVALUATION",
        "Two independent blockers: the weights do not match the documented model, and the evaluator points to missing private assets.",
        width,
        theme,
    )
    widths = _fit_widths(width, (0.22, 0.28, 0.50), (15, 19, 31))
    lines.extend(
        _table(
            ("Probe", "Observed result", "Interpretation"),
            (
                (
                    "README UNet load",
                    Cell("0 / 154 keys match", "bad"),
                    "The documented UNet receives none of the released parameters; strict=False hides the mismatch.",
                ),
                (
                    "Released chunks",
                    "115 tensors reconstructed",
                    "All 115 match only part of the PPO actor-critic encoder.",
                ),
                (
                    "Policy completeness",
                    Cell("25 parameters absent", "bad"),
                    "Missing encoder fusion, all actor heads, all critic heads, and actor_log_std.",
                ),
                (
                    "Evaluator checkpoint",
                    Cell("Missing private path", "bad"),
                    "FYP/logs_bk/run_20250211_003345/final_model.pth",
                ),
                (
                    "Evaluator interface",
                    Cell("README flags ignored", "bad"),
                    "No argument parser; omitted dependencies and hard-coded FYP, /raid/candi, and E:/Study/UCL paths.",
                ),
            ),
            widths,
            theme,
        )
    )
    lines.extend(
        [
            "",
            *_callout(
                "RELEASE",
                "The files are neither a usable surrogate UNet nor a complete PPO policy checkpoint. The advertised evaluation cannot reach a paper-matching inference run from a clean environment.",
                width,
                theme,
                "bad",
            ),
            *_callout(
                "LIMIT",
                "This is an artifact-availability and compatibility finding. It does not reveal whether a complete private checkpoint once existed.",
                width,
                theme,
                "limit",
            ),
        ]
    )
    return lines


def _manuscript_code(width: int, theme: Theme) -> list[str]:
    lines = _banner(
        "5. MANUSCRIPT-TO-CODE DISCREPANCIES",
        "Training the current public code would test a materially different method, not reproduce the manuscript method.",
        width,
        theme,
    )
    widths = _fit_widths(width, (0.20, 0.32, 0.34, 0.14), (14, 21, 22, 10))
    lines.extend(
        _table(
            ("Component", "Manuscript", "Public code", "Finding"),
            (
                (
                    "Agent state",
                    Cell("Four channels: 3 MR + current mask", "claim"),
                    "Six channels: 3 MR + current, entropy, and history masks",
                    Cell("Contradicted", "bad"),
                ),
                (
                    "Initial prompt",
                    Cell("Random point inside lesion", "claim"),
                    "Ground-truth distance-transform center-region prompt",
                    Cell("Discrepant", "warn"),
                ),
                (
                    "Mask transition",
                    Cell("New region replaces old mask", "claim"),
                    "Logical union / accumulation",
                    Cell("Contradicted", "bad"),
                ),
                (
                    "Termination",
                    Cell("Stable mask or maximum T", "claim"),
                    "terminated=False; only maximum-step truncation",
                    Cell("Contradicted", "bad"),
                ),
                (
                    "Reward",
                    Cell("Dice improvement + beta x mean entropy", "claim"),
                    "Adaptive IoU, Dice, streak, entropy, and boundary terms",
                    Cell("Contradicted", "bad"),
                ),
                (
                    "Region growing",
                    Cell("Local intensity SD + entropy cutoffs", "claim"),
                    "Weighted modality-distance rule, entropy, and size constraints",
                    Cell("Contradicted", "bad"),
                ),
                (
                    "Architecture",
                    Cell("GroupNorm + LeakyReLU + three FC layers", "claim"),
                    "BatchNorm / ReLU / residual attention; LayerNorm / ReLU / Dropout heads",
                    Cell("Contradicted", "bad"),
                ),
                (
                    "Split / config",
                    Cell("80:20; supplement PPO settings", "claim"),
                    "70:15:15 plus materially different YAML and runtime overrides",
                    Cell("Contradicted", "bad"),
                ),
            ),
            widths,
            theme,
        )
    )
    lines.extend(
        [
            "",
            *_callout(
                "CONTEXT",
                "The paper openly discloses that simulated inference prompts are sampled inside the ground-truth lesion. That is not hidden. The code is still more center-biased than the stated random simulation.",
                width,
                theme,
                "info",
            ),
            *_callout(
                "CONSEQUENCE",
                "Architecture, preprocessing, reward, prompt generation, region exploration, and published hyperparameters must be version-matched before a retraining result can be called a reproduction.",
                width,
                theme,
                "warn",
            ),
        ]
    )
    return lines


def _statistics(width: int, theme: Theme) -> list[str]:
    lines = _banner(
        "6. STATISTICAL AND REPORTING DISCREPANCIES",
        "We tested whether the published paired-test p-values could follow from the reported summaries and holdout sizes.",
        width,
        theme,
    )
    widths = _fit_widths(width, (0.43, 0.17, 0.20, 0.20), (27, 11, 14, 14))
    lines.extend(
        _table(
            ("Comparison", "Reported p", "Largest possible p", "Largest fitting subset n"),
            (
                ("PROMIS RL vs Swin, n=114", Cell("0.010", "claim"), Cell("0.0006419", "bad"), "65"),
                ("PROMIS RL vs UniverSeg, n=114", Cell("0.002", "claim"), Cell("1.334e-10", "bad"), "26"),
                ("PI-CAI RL vs Swin, implied n=218", Cell("0.004", "claim"), Cell("3.545e-6", "bad"), "84"),
                ("PI-CAI RL vs UniverSeg, implied n=218", Cell("0.001", "claim"), Cell("7.086e-22", "bad"), "26"),
            ),
            widths,
            theme,
        )
    )
    lines.extend(
        [
            "",
            *_callout(
                "METHOD",
                "For paired observations, SD(X-Y) cannot exceed SD(X)+SD(Y). Even choosing rounding in the paper's favor gives the upper bounds above.",
                width,
                theme,
                "evidence",
            ),
            *_callout(
                "INFERENCE",
                "The four p-values cannot be paired tests on every stated/implied holdout case under the published summaries. A smaller undisclosed subset, a different test, or a reporting/analysis error could explain this; the cause is unverifiable without per-case data and statistics code.",
                width,
                theme,
                "warn",
            ),
            *_callout(
                "TIME CLAIM",
                "1093 seconds / 131 seconds = 8.3435x, not 10x. That is still an 88.0% reduction, but the tenfold wording is arithmetically overstated.",
                width,
                theme,
                "warn",
            ),
            *_callout(
                "HUMAN CLAIM",
                "p=0.14 is failure to detect a difference, not evidence of equivalence. No equivalence/noninferiority margin or confidence interval was reported, and the comparison used one reader on PROMIS only.",
                width,
                theme,
                "warn",
            ),
        ]
    )
    return lines


def _compute(width: int, theme: Theme) -> list[str]:
    lines = _banner(
        "7. WOULD AN H200 HAVE CHANGED THE RESULT?",
        "No. More compute changes runtime or experiment scale; it does not change a frozen evaluation or restore missing artifacts.",
        width,
        theme,
    )
    widths = _fit_widths(width, (0.30, 0.32, 0.38), (20, 21, 24))
    lines.extend(
        _table(
            ("Question", "Answer", "Reason"),
            (
                ("Re-run sealed policy test", Cell("Same scientific verdict", "bad"), "Same frozen inputs, actions, and metrics; workload used about 4 GB GPU memory."),
                ("Run paper evaluator", Cell("Still blocked", "bad"), "The checkpoint, exact splits, evaluator, and holdout outputs are missing; this is not a compute shortage."),
                ("Train a redesigned method", Cell("Potentially faster", "warn"), "H200 can accelerate native-resolution, multi-fold, multi-seed training on a larger cohort."),
                ("Claim the old method worked", Cell("No", "bad"), "A new, larger experiment would test a redesigned method, not rescue the failed frozen policy."),
            ),
            widths,
            theme,
        )
    )
    lines.extend(
        [
            "",
            *_callout(
                "H200 GATE",
                "Rent high-end compute only after a frozen paper-matching protocol, exact split manifests, complete code/checkpoints, and a predeclared multi-seed evaluation make retraining scientifically interpretable.",
                width,
                theme,
                "warn",
            )
        ]
    )
    return lines


def _decision(width: int, theme: Theme) -> list[str]:
    lines = _banner(
        "8. FINAL PI DECISION",
        "Two negative findings, each with a different and explicit claim boundary.",
        width,
        theme,
    )
    lines.extend(
        [
            theme.paint("SEGMENTATION POLICY", "heading"),
            *_bullets(
                (
                    Cell("Keep ResEnc-L / KEEP as the current segmentation policy.", "good"),
                    Cell("The sealed hybrid changed one study and harmed it; the broader screen harmed all three changed studies.", "bad"),
                    Cell("Do not claim RL/EDL improvement and do not launch PPO from this formulation.", "bad"),
                    "A future study needs a redesigned state/action/reward formulation, more independent patients, and external validation.",
                ),
                width,
                theme,
            ),
            "",
            theme.paint("PAPER REPRODUCTION", "heading"),
            *_bullets(
                (
                    Cell("Do not treat the paper as independently validated evidence that RL improves segmentation.", "bad"),
                    "Request the exact commit, complete checkpoints, split manifests, preprocessing, paper-matching evaluator, per-case outputs, and statistics code.",
                    Cell("The release has serious reproducibility, statistical, and implementation discrepancies.", "warn"),
                    Cell("The evidence does not establish fabrication or deliberate deception; intent cannot be inferred from missing or inconsistent artifacts.", "limit"),
                ),
                width,
                theme,
            ),
            "",
            *_callout(
                "BOTTOM LINE",
                "Our tested policies did not safely enhance segmentation, and the source paper's linked public release cannot reproduce or verify its headline performance. The scientifically defensible next move is KEEP plus an author-material request, not more GPU time on the current formulations.",
                width,
                theme,
                "bad",
            ),
        ]
    )
    return lines


SECTIONS: tuple[Section, ...] = (
    Section("pipeline", "Pipeline timeline", "From baseline through the sealed test and paper audit.", _pipeline),
    Section("policy", "Sealed RL/EDL result", "Dice, confidence interval, coverage, harm, and decision.", _policy_result),
    Section("paper", "Paper claims vs release", "Which headline claims can and cannot be verified.", _paper_claims),
    Section("release", "Weight and evaluator forensics", "0/154 match and the private checkpoint blocker.", _release_forensics),
    Section("code", "Manuscript vs code", "Eight audited implementation and configuration discrepancies.", _manuscript_code),
    Section("statistics", "Statistics and reporting", "P-value bounds, 8.34x timing, and the human comparison.", _statistics),
    Section("compute", "H200 decision", "Why faster hardware would not change the frozen result.", _compute),
    Section("decision", "Final PI decision", "What we can conclude and what should happen next.", _decision),
)
SECTION_BY_KEY = {section.key: section for section in SECTIONS}


def render_section(key: str, *, width: int = 104, color: bool = False) -> str:
    """Return one deterministic section, primarily for tests and recording prep."""

    try:
        section = SECTION_BY_KEY[key]
    except KeyError as error:
        raise ValueError(f"unknown section: {key}") from error
    width = max(80, min(width, 132))
    return "\n".join(section.renderer(width, Theme(color)))


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or greater")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--list", action="store_true", help="list presentation sections and exit")
    parser.add_argument(
        "--section",
        action="append",
        choices=tuple(SECTION_BY_KEY),
        help="show only this section; repeat to select multiple sections",
    )
    parser.add_argument(
        "--mode",
        choices=("interactive", "auto"),
        help="wait for Enter or advance automatically (default follows terminal interactivity)",
    )
    parser.add_argument("--delay", type=_positive_float, default=1.25, help="seconds between sections in auto mode")
    parser.add_argument("--no-color", action="store_true", help="disable ANSI styling")
    parser.add_argument("--width", type=int, help="presentation width, clamped to 80-132 columns")
    return parser.parse_args(argv)


def _is_tty(stream: TextIO) -> bool:
    isatty = getattr(stream, "isatty", None)
    return bool(isatty and isatty())


def _list_sections(stream: TextIO, theme: Theme) -> None:
    print(theme.paint("AVAILABLE SECTIONS", "heading"), file=stream)
    for section in SECTIONS:
        print(f"  {theme.paint(section.key.ljust(12), 'info')} {section.summary}", file=stream)


def main(
    argv: Sequence[str] | None = None,
    *,
    stream: TextIO | None = None,
    input_fn: Callable[[str], str] = input,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    """Run the walkthrough and return a process exit status."""

    args = _parse_args(argv)
    output = stream or sys.stdout
    tty = _is_tty(output)
    theme = Theme(enabled=tty and not args.no_color)
    if args.list:
        _list_sections(output, theme)
        return 0

    selected = [SECTION_BY_KEY[key] for key in args.section] if args.section else list(SECTIONS)
    terminal_width = shutil.get_terminal_size((104, 30)).columns
    width = max(80, min(args.width or terminal_width, 132))
    mode = args.mode or ("interactive" if tty else "auto")

    print("\n".join(_opening(width, theme)), file=output)
    for index, section in enumerate(selected):
        print(render_section(section.key, width=width, color=theme.enabled), file=output)
        output.flush()
        if index == len(selected) - 1:
            continue
        if mode == "interactive":
            response = input_fn(theme.paint("[Enter] next section | q quit: ", "muted"))
            print("", file=output)
            if response.strip().lower() in {"q", "quit", "exit"}:
                print(theme.paint("Walkthrough stopped by presenter.", "warn"), file=output)
                return 0
        elif args.delay:
            sleep_fn(args.delay)
            print("", file=output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
