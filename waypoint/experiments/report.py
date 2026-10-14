"""Write a batch's results: CSVs, a JSON summary, a console table, and plots.

    per_mission.csv   one row per flight pair (condition x mission x repeat)
    replans.csv       one row per in-flight replan trigger
    summary.json      the aggregates, per condition
    catch_rate.png    verifier catch rate by condition
    replan_latency.png  trigger-to-verified-plan latency by fault type
    replan_outcomes.png how in-flight replans resolved, by condition
    tradeoff.png      verified vs unconstrained path length, per condition

Plots need matplotlib (`pip install -e ".[experiments]"`); without it the
CSVs and the summary are still written and the plots are skipped with a note.
"""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

from waypoint.experiments.metrics import (
    MissionRow,
    ReplanRow,
    aggregate,
    as_dict,
    ordered_conditions,
    row_fields,
)
from waypoint.experiments.spec import CONDITION_LABELS

# The reference data-viz palette: categorical slots in fixed order, so a
# condition keeps its colour in every chart and every run.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SLOTS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948")
#: Ordinal blue ramp (dark to light) for attempts, and status critical for abandoned.
OUTCOME_COLOURS = {
    "first_try": "#184f95",
    "second_try": "#3987e5",
    "later": "#86b6ef",
    "abandoned": "#d03b3b",
}
OUTCOME_LABELS = {
    "first_try": "passed 1st try",
    "second_try": "passed 2nd try",
    "later": "passed 3rd+ try",
    "abandoned": "abandoned",
}


def colour(condition: str) -> str:
    """A condition's colour is its place in the fixed order, never its rank in a chart."""
    if condition in CONDITION_LABELS:
        return SLOTS[CONDITION_LABELS.index(condition)]
    return MUTED


def write_report(
    rows: list[MissionRow],
    replans: list[ReplanRow],
    out_dir: Path,
    *,
    plots: bool = True,
    echo=print,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = aggregate(rows, replans)
    _write_csv(out_dir / "per_mission.csv", row_fields(MissionRow), [as_dict(r) for r in rows])
    _write_csv(
        out_dir / "replans.csv",
        [*row_fields(ReplanRow), "bucket"],
        [{**as_dict(r), "bucket": r.bucket} for r in replans],
    )
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    echo(format_summary(summary))
    if plots:
        written = make_plots(rows, replans, summary, out_dir)
        if written is None:
            echo('\n(plots skipped: matplotlib is not installed; pip install -e ".[experiments]")')
        else:
            echo("\n" + "\n".join(f"  wrote {path}" for path in written))
    echo(f"  wrote {out_dir / 'per_mission.csv'}, replans.csv, summary.json")
    return summary


def _write_csv(path: Path, columns: list[str], records: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for record in records:
            writer.writerow({k: ("" if v is None else v) for k, v in record.items()})


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


def _share(stats: dict) -> str:
    cost = stats.get("cost") or {}
    return f"{cost['tokens_cached'] / cost['tokens_in']:.0%}" if cost.get("tokens_in") else "-"


def _interval(bounds) -> str:
    return "-" if not bounds else f"{bounds[0] * 100:.0f} to {bounds[1] * 100:.0f}%"


def _pooled(first_plan: dict | None) -> str:
    if not first_plan or first_plan["pooled_catch_rate"] is None:
        return "-"
    return f"{first_plan['pooled_catch_rate'] * 100:.0f}% ({first_plan['caught']}/{first_plan['planned']})"


def format_summary(summary: dict) -> str:
    conditions = list(summary)
    if not conditions:
        return "No flights to summarise."

    def pct(value):
        return "-" if value is None else f"{value * 100:.0f}%"

    def num(value, unit=""):
        return "-" if value is None else f"{value:.2f}{unit}"

    def of(stats, path):
        for key in path.split("."):
            stats = stats.get(key) if isinstance(stats, dict) else None
        return stats

    lines = [
        ("flights (errors, planning failed)", lambda s: f"{s['flights']} ({s['errors']}, {s['planning_failed']})"),
        ("verifier catch rate", lambda s: f"{pct(s['catch_rate'])} ({s['caught']}/{s['planned']})"),
        ("  95% interval", lambda s: _interval(s.get("catch_rate_ci95"))),
        ("  pooled, same first prompt", lambda s: _pooled(s.get("first_plan"))),
        ("  any finding, warnings too", lambda s: pct(s["any_violation_rate"])),
        ("  caught, then corrected", lambda s: pct(s["ground_correction_rate"])),
        ("completed", lambda s: f"{pct(s['completion_rate'])} ({s['completed']}/{s['planned']})"),
        ("crashed / broke a keep-out", lambda s: f"{pct(s['crash_rate'])} / {pct(s['separation_breach_rate'])}"),
        ("in-flight replans triggered", lambda s: str(of(s, "replans.triggered"))),
        ("  passed 1st try", lambda s: pct(of(s, "replans.first_try_rate"))),
        ("  passed within 2 tries", lambda s: pct(of(s, "replans.within_two_rate"))),
        ("  abandoned", lambda s: pct(of(s, "replans.abandoned_rate"))),
        ("replan latency, median", lambda s: num(of(s, "latency_s.median"), " s")),
        ("verified vs baseline, pairs", lambda s: str(of(s, "tradeoff.pairs"))),
        ("  distance overhead, median", lambda s: num(of(s, "tradeoff.median_distance_overhead_pct"), "%")),
        ("  time overhead, median", lambda s: num(of(s, "tradeoff.median_time_overhead_pct"), "%")),
        ("baseline crashed", lambda s: pct(of(s, "tradeoff.baseline_crash_rate"))),
        ("baseline broke a keep-out", lambda s: pct(of(s, "tradeoff.baseline_separation_breach_rate"))),
        ("baseline flew a refused plan", lambda s: pct(of(s, "tradeoff.baseline_unsafe_plan_rate"))),
        ("planner calls", lambda s: str(of(s, "cost.llm_calls"))),
        ("  output tokens per flight pair", lambda s: num(of(s, "cost.tokens_out_per_flight"))),
        ("  input tokens (cached)", lambda s: f"{of(s, 'cost.tokens_in') or 0:,} ({_share(s)})"),
    ]
    width = max(len(name) for name, _ in lines) + 2
    columns = [max(len(c), 14) for c in conditions]
    header = " " * width + "  ".join(c.rjust(w) for c, w in zip(conditions, columns))
    body = [header, "-" * len(header)]
    for name, render in lines:
        cells = [render(summary[c]).rjust(w) for c, w in zip(conditions, columns)]
        body.append(name.ljust(width) + "  ".join(cells))
    if any(len(summary[c].get("first_plan", {}).get("conditions", ())) > 1 for c in conditions):
        body += [
            "",
            "note: violation feedback only changes replans, so feedback on and off share a",
            "first-plan prompt; a catch-rate gap between them is sampling noise. Compare the",
            "pooled rate across hierarchical and flat instead.",
        ]

    faults = sorted({f for c in conditions for f in summary[c]["latency_s"]["by_fault"]})
    if faults:
        body += ["", "replan latency by fault type (median s, n):"]
        for fault in faults:
            cells = []
            for c, w in zip(conditions, columns):
                stats = summary[c]["latency_s"]["by_fault"].get(fault)
                cells.append(("-" if not stats else f"{stats['median']:.2f} ({stats['n']})").rjust(w))
            body.append(("  " + fault).ljust(width) + "  ".join(cells))

    models = sorted({m for c in conditions for m in (summary[c].get("cost") or {}).get("models", {})})
    if models:
        body += ["", "which models answered (flight pairs that used each):"]
        for model in models:
            cells = [str(summary[c]["cost"]["models"].get(model, 0)).rjust(w) for c, w in zip(conditions, columns)]
            body.append(("  " + model).ljust(width) + "  ".join(cells))

    kinds = sorted({k for c in conditions for k in summary[c]["violation_types"]})
    if kinds:
        body += ["", "what the verifier caught in first plans (missions):"]
        for kind in kinds:
            cells = [str(summary[c]["violation_types"].get(kind, 0)).rjust(w) for c, w in zip(conditions, columns)]
            body.append(("  " + kind).ljust(width) + "  ".join(cells))
    return "\n".join(body)


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(rows, replans, summary, out_dir: Path) -> list[Path] | None:
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        return None

    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.edgecolor": AXIS,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
    })
    conditions = list(summary)
    written = []
    for name, draw in (
        ("catch_rate.png", _plot_catch_rate),
        ("replan_latency.png", _plot_latency),
        ("replan_outcomes.png", _plot_outcomes),
        ("tradeoff.png", _plot_tradeoff),
    ):
        fig = draw(plt, conditions, rows, replans, summary)
        if fig is None:
            continue
        path = out_dir / name
        fig.savefig(path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
    return written


def _plot_catch_rate(plt, conditions, rows, replans, summary):
    fig, ax = plt.subplots(figsize=(max(4.5, 1.6 * len(conditions) + 1.5), 3.8))
    rates = [(summary[c]["catch_rate"] or 0.0) * 100 for c in conditions]
    bars = ax.bar(range(len(conditions)), rates, width=0.6, color=SLOTS[0], edgecolor=SURFACE, linewidth=2)
    for i, (bar, c) in enumerate(zip(bars, conditions)):
        s = summary[c]
        top = bar.get_height()
        bounds = s.get("catch_rate_ci95")
        if bounds:
            low, high = bounds[0] * 100, bounds[1] * 100
            ax.plot([i, i], [low, high], color=INK_2, linewidth=1.5, zorder=3,
                    label="95% interval")
            top = high
        ax.annotate(
            f"{(s['catch_rate'] or 0) * 100:.0f}%  ({s['caught']}/{s['planned']})",
            (bar.get_x() + bar.get_width() / 2, top),
            xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", color=INK, fontsize=9,
        )
    ax.set_xticks(range(len(conditions)), conditions)
    ax.set_ylim(0, 105)
    ax.set_ylabel("first plans rejected (%)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Verifier catch rate by condition")
    if any(len(summary[c].get("first_plan", {}).get("conditions", ())) > 1 for c in conditions):
        fig.text(
            0.01, -0.02,
            "Feedback on/off share a first-plan prompt: gaps between them are sampling noise.",
            color=INK_2, fontsize=8.5, ha="left", va="top",
        )
    if any(summary[c].get("catch_rate_ci95") for c in conditions):
        _dedupe_legend(ax, loc="upper right")
    return fig


def _plot_latency(plt, conditions, rows, replans, summary):
    timed = [r for r in replans if r.latency_s is not None]
    if not timed:
        return None
    faults = sorted({r.fault_type for r in timed})
    present = [c for c in conditions if any(r.condition == c for r in timed)]
    fig, ax = plt.subplots(figsize=(max(5.0, 1.8 * len(faults) + 2), 4.0))
    span = 0.7
    step = span / max(1, len(present))
    for k, condition in enumerate(present):
        offset = -span / 2 + step * (k + 0.5)
        for i, fault in enumerate(faults):
            values = sorted(r.latency_s for r in timed if r.condition == condition and r.fault_type == fault)
            if not values:
                continue
            xs = [i + offset + (j - (len(values) - 1) / 2) * min(0.02, step / (len(values) + 1)) for j in range(len(values))]
            ax.scatter(xs, values, s=36, color=colour(condition), edgecolor=SURFACE, linewidth=1.2, zorder=3,
                       label=condition)
            middle = statistics.median(values)
            ax.plot([i + offset - step * 0.35, i + offset + step * 0.35], [middle, middle], color=INK_2, linewidth=2, zorder=4)
    ax.set_xticks(range(len(faults)), [f.replace("_", " ").lower() for f in faults])
    ax.set_xlim(-0.6, len(faults) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("trigger to verified plan (s)")
    ax.grid(axis="x", visible=False)
    ax.set_title("Replan latency by fault type (bar = median)")
    if len(present) > 1:
        _dedupe_legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    else:
        ax.set_title(f"Replan latency by fault type (bar = median), {present[0]}")
    return fig


def _plot_outcomes(plt, conditions, rows, replans, summary):
    present = [c for c in conditions if summary[c]["replans"]["triggered"]]
    if not present:
        return None
    fig, ax = plt.subplots(figsize=(7.0, 0.7 * len(present) + 1.6))
    for y, condition in enumerate(present):
        stats = summary[condition]["replans"]
        left = 0.0
        for key in OUTCOME_COLOURS:
            share = stats[key] / stats["triggered"] * 100
            if not share:
                continue
            ax.barh(y, share, left=left, height=0.55, color=OUTCOME_COLOURS[key], edgecolor=SURFACE, linewidth=2,
                    label=OUTCOME_LABELS[key])
            if share >= 9:
                text = SURFACE if key in ("first_try", "second_try", "abandoned") else INK
                ax.text(left + share / 2, y, f"{stats[key]}", ha="center", va="center", color=text, fontsize=9)
            left += share
        ax.text(101, y, f"n={stats['triggered']}", va="center", color=INK_2, fontsize=9)
    ax.set_yticks(range(len(present)), present)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("in-flight replans (%)")
    ax.grid(axis="y", visible=False)
    ax.set_title("How in-flight replans resolved")
    _dedupe_legend(ax, loc="upper center", bbox_to_anchor=(0.5, -0.12 - 0.35 / (len(present) + 1)), ncol=4)
    return fig


def _plot_tradeoff(plt, conditions, rows, replans, summary):
    present = [c for c in conditions if any(r.condition == c and r.distance_overhead_pct is not None for r in rows)]
    if not present:
        return None
    fig, axes = plt.subplots(1, len(present), figsize=(3.6 * len(present) + 0.6, 3.8), squeeze=False)
    for ax, condition in zip(axes[0], present):
        pairs = [r for r in rows if r.condition == condition and r.distance_overhead_pct is not None]
        xs = [r.baseline_distance_m for r in pairs]
        ys = [r.distance_m for r in pairs]
        top = max(xs + ys) * 1.08
        ax.plot([0, top], [0, top], color=AXIS, linewidth=1.5, linestyle="--", zorder=1)
        ax.scatter(xs, ys, s=40, color=colour(condition), edgecolor=SURFACE, linewidth=1.2, zorder=3)
        median = summary[condition]["tradeoff"]["median_distance_overhead_pct"]
        ax.set_title(f"{condition}\nmedian {median:+.1f}% distance" if median is not None else condition, fontsize=10)
        ax.set_xlim(0, top)
        ax.set_ylim(0, top)
        ax.set_aspect("equal")
        ax.set_xlabel("unconstrained baseline (m)")
    axes[0][0].set_ylabel("verified run (m)")
    fig.suptitle("Path length: what safety costs (dashed = no cost)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    return fig


def _dedupe_legend(ax, **kwargs) -> None:
    handles, labels = ax.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for handle, label in zip(handles, labels):
        if label and label not in seen:
            seen[label] = handle
    ax.legend(list(seen.values()), list(seen), **kwargs)
