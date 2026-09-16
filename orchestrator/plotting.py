"""Render a mission run as an interactive 3D Plotly figure.

The flown path is coloured by remaining battery along its length; the planned
waypoints sit on top as labelled markers so planned and actual can be compared
directly.
"""

from __future__ import annotations

import webbrowser
from pathlib import Path

import plotly.graph_objects as go

# Marker symbol and colour per waypoint action. Scatter3d supports a small
# symbol set, so these are chosen from what it actually renders.
ACTION_STYLE = {
    "MOVE": ("circle", "#4C78A8"),
    "HOVER": ("square", "#F58518"),
    "CAPTURE": ("diamond", "#54A24B"),
    "LAND": ("x", "#E45756"),
}
_DEFAULT_STYLE = ("circle-open", "#888888")

BATTERY_COLORSCALE = "Turbo"


def build_figure(run: dict) -> go.Figure:
    """Build the 3D figure for a run dict loaded from a run file."""
    ticks = run.get("ticks") or []
    if not ticks:
        raise ValueError(
            "This run has no telemetry ticks, so there is no flight path to plot. "
            "(A mission that failed verification never flies.)"
        )

    fig = go.Figure()
    _add_planned_route(fig, run.get("waypoints") or [])
    _add_flown_path(fig, ticks)
    _add_waypoint_markers(fig, run.get("waypoints") or [])
    _style(fig, run)
    return fig


def _add_flown_path(fig: go.Figure, ticks: list[dict]) -> None:
    battery = [t["battery_pct"] for t in ticks]
    # One row per tick, in the order the hovertemplate reads them.
    customdata = [
        [t["timestamp"], t["velocity"], t["heading"], t["battery_pct"]] for t in ticks
    ]
    fig.add_trace(
        go.Scatter3d(
            x=[t["x"] for t in ticks],
            y=[t["y"] for t in ticks],
            z=[t["z"] for t in ticks],
            mode="lines+markers",
            name="Flown path",
            line=dict(color=battery, colorscale=BATTERY_COLORSCALE, width=6),
            marker=dict(
                size=2.5,
                color=battery,
                colorscale=BATTERY_COLORSCALE,
                cmin=0,
                cmax=100,
                colorbar=dict(title="Battery %", x=1.02),
            ),
            customdata=customdata,
            hovertemplate=(
                "<b>t = %{customdata[0]:.1f} s</b><br>"
                "pos (%{x:.1f}, %{y:.1f}, %{z:.1f}) m<br>"
                "velocity %{customdata[1]:.2f} m/s<br>"
                "heading %{customdata[2]:.1f}&#176;<br>"
                "battery %{customdata[3]:.1f}%"
                "<extra></extra>"
            ),
        )
    )


def _add_planned_route(fig: go.Figure, waypoints: list[dict]) -> None:
    """The planned polyline, drawn faintly underneath the flown path."""
    if not waypoints:
        return
    fig.add_trace(
        go.Scatter3d(
            x=[w["x"] for w in waypoints],
            y=[w["y"] for w in waypoints],
            z=[w["z"] for w in waypoints],
            mode="lines",
            name="Planned route",
            line=dict(color="rgba(130,130,130,0.55)", width=3, dash="dash"),
            hoverinfo="skip",
        )
    )


def _add_waypoint_markers(fig: go.Figure, waypoints: list[dict]) -> None:
    """One trace per action, so the legend doubles as an action filter."""
    for action in dict.fromkeys(w["action"] for w in waypoints):
        group = [w for w in waypoints if w["action"] == action]
        symbol, color = ACTION_STYLE.get(action, _DEFAULT_STYLE)
        fig.add_trace(
            go.Scatter3d(
                x=[w["x"] for w in group],
                y=[w["y"] for w in group],
                z=[w["z"] for w in group],
                mode="markers+text",
                name=action,
                marker=dict(size=6, symbol=symbol, color=color, line=dict(width=1)),
                text=[w["subtask"] for w in group],
                textposition="top center",
                textfont=dict(size=9, color=color),
                customdata=[[w["subtask"], w["index"], w["speed"]] for w in group],
                hovertemplate=(
                    f"<b>{action}</b><br>"
                    "subtask %{customdata[0]}<br>"
                    "waypoint #%{customdata[1]}<br>"
                    "pos (%{x:.1f}, %{y:.1f}, %{z:.1f}) m<br>"
                    "planned speed %{customdata[2]:.1f} m/s"
                    "<extra></extra>"
                ),
            )
        )


def _style(fig: go.Figure, run: dict) -> None:
    ticks = run["ticks"]
    duration = ticks[-1]["timestamp"] if ticks else 0.0
    end_battery = ticks[-1]["battery_pct"] if ticks else 0.0
    subtitle = (
        f"{len(ticks)} ticks &middot; {duration:.0f} s &middot; "
        f"{len(run.get('waypoints') or [])} waypoints &middot; "
        f"battery {end_battery:.1f}% at touchdown"
    )
    fig.update_layout(
        title=dict(
            text=f"<b>{run.get('mission_text', 'Mission')}</b><br>"
            f"<span style='font-size:12px;color:#666'>{subtitle}</span>"
        ),
        scene=dict(
            xaxis_title="x — east (m)",
            yaxis_title="y — north (m)",
            zaxis_title="z — altitude (m)",
            # Equal scaling, or the altitude axis stretches and the path lies.
            aspectmode="data",
        ),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, t=70, b=0),
        template="plotly_white",
        hovermode="closest",
    )


def html_path_for(run_file: Path | str) -> Path:
    """The HTML file that sits beside its run file."""
    return Path(run_file).with_suffix(".html")


def render(
    run: dict,
    run_file: Path | str | None,
    *,
    open_browser: bool = True,
) -> Path:
    """Write the standalone HTML next to the run file and optionally open it."""
    # Built first so an unplottable run fails with a clear message before we
    # touch the filesystem or try to resolve a path that may be None.
    figure = build_figure(run)
    if run_file is None:
        raise ValueError("This run was never written to a file, so there is nowhere to save the plot.")
    out = html_path_for(run_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    # include_plotlyjs="inline" keeps the file self-contained and offline-usable.
    figure.write_html(str(out), include_plotlyjs="inline", full_html=True)
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return out
