#!/usr/bin/env python3
"""Draw the README figures from the tables in results/ (written by export_results.py).

Needs only numpy and matplotlib (project 1's environment has both):

    ~/nexarm-teleop-latency/.venv/bin/python scripts/make_figures.py

Outputs: docs/img/success_vs_time.png, docs/img/control_timing.png, docs/img/placement_coverage.png
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = ROOT / "docs" / "img"

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e6e5e1"
REFERENCE = "#a3a19a"  # earlier baseline (ACT, 45 demos) and training demos
# Colour follows the policy in every figure: categorical slots 1-3 in fixed order.
COLOR = {"act102": "#2a78d6", "sync": "#eb6834", "rtc": "#1baf7a"}

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "text.color": TEXT, "axes.labelcolor": TEXT_2,
        "xtick.color": TEXT_2, "ytick.color": TEXT_2, "axes.edgecolor": GRID,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 1.0,
        "axes.axisbelow": True, "xtick.major.size": 0, "ytick.major.size": 0,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False,
    }
)


def read(name: str) -> list[dict]:
    with (RES / name).open(newline="") as f:
        return list(csv.DictReader(f))


def releases(trials: list[dict], session: str) -> tuple[list[float], int]:
    """Release times of the trials the operator scored as successes, and the number of trials."""
    rows = [r for r in trials if r["session"] == session]
    t = sorted(float(r["release_over_bin_s"]) for r in rows if r["operator_result"] == "success")
    return t, len(rows)


def cumulative(ax, t: list[float], n: int, color: str, label: str, dashed: bool = False) -> None:
    x = [0.0, *t, 40.0]
    y = [*range(len(t) + 1), len(t)]
    ax.step(x, y, where="post", color=color, lw=2, ls="--" if dashed else "-",
            label=f"{label}: {len(t)}/{n}, median {np.median(t):.1f} s")


def success_vs_time() -> None:
    trials = read("trials.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True, constrained_layout=True)
    ax = axes[0]
    cumulative(ax, *releases(trials, "act45_100k_ab"), REFERENCE, "ACT, 45 demos", dashed=True)
    cumulative(ax, *releases(trials, "act102_100k_ab"), COLOR["act102"], "ACT, 102 demos")
    ax.set_title("More demonstrations (ACT, Nov 17, interleaved)")
    ax.set_ylabel("Placements with the box in the bin (of 20)")
    ax = axes[1]
    cumulative(ax, *releases(trials, "act102_100k_1118"), COLOR["act102"], "ACT, 102 demos")
    cumulative(ax, *releases(trials, "pi05_12k_smoke"), COLOR["sync"], "π0.5, synchronous")
    cumulative(ax, *releases(trials, "pi05_12k_rtc"), COLOR["rtc"], "π0.5, RTC")
    ax.set_title("ACT vs π0.5, same 102 demos (Nov 18, same session)")
    for ax in axes:
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 20.5)
        ax.set_yticks([0, 5, 10, 15, 20])
        ax.set_xlabel("Seconds since the policy started")
        ax.legend(loc="upper left", handlelength=2.4)
    fig.savefig(OUT / "success_vs_time.png", dpi=150)
    plt.close(fig)


def control_timing() -> None:
    rows = read("frame_intervals.csv")
    panels = [("pi05_12k_smoke", "sync", "π0.5, synchronous: re-plans in the control loop"),
              ("pi05_12k_rtc", "rtc", "π0.5, RTC: re-plans in a background thread"),
              ("act102_100k_1118", "act102", "ACT, 102 demos: re-plans in the control loop")]
    fig, axes = plt.subplots(3, 1, figsize=(11, 5.6), sharex=True, constrained_layout=True)
    for ax, (session, key, title) in zip(axes, panels):
        r = [x for x in rows if x["session"] == session]
        t = np.array([float(x["t_s"]) for x in r])
        dt = np.array([float(x["interval_ms"]) for x in r])
        hz = len(dt) / (t[-1] + dt[-1] / 1000 - t[0])
        ax.axhline(1000 / 15, color=TEXT_2, lw=1, ls=(0, (4, 3)))
        ax.plot(t, dt, color=COLOR[key], lw=1.5, solid_joinstyle="round")
        ax.set_title(f"{title}  ·  {hz:.1f} Hz, {(n := int((dt > 150).sum()))} gap{"" if n == 1 else "s"} over 150 ms", fontsize=10.5)
        ax.set_ylim(0, 620)
        ax.set_yticks([0, 200, 400, 600])
        ax.set_ylabel("Frame gap (ms)")
    axes[1].annotate("66.7 ms = 15 Hz target", (40, 1000 / 15), textcoords="offset points", xytext=(0, 8),
                     ha="right", va="bottom", color=TEXT_2, fontsize=9)
    axes[-1].set_xlim(0, 40)
    axes[-1].set_xlabel("Seconds since the policy started (fixed placement 6, one trial each)")
    fig.savefig(OUT / "control_timing.png", dpi=150)
    plt.close(fig)


def placement_coverage() -> None:
    rows = read("grasp_poses.csv")
    demo = [r for r in rows if r["source"] == "training demo"]
    rand = [r for r in rows if r["source"] == "random placement"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    ax.grid(axis="x", color=GRID, lw=1)

    def xy(rs):
        return [float(r["azimuth_deg"]) for r in rs], [float(r["orientation_deg"]) for r in rs]

    ax.scatter(*xy(demo), s=22, color=REFERENCE, lw=0, label=f"Training demos ({len(demo)})", zorder=2)
    ok = [r for r in rand if r["operator_result"] == "success"]
    bad = [r for r in rand if r["operator_result"] == "fail"]
    ax.scatter(*xy(ok), s=70, color=COLOR["rtc"], edgecolors=SURFACE, linewidths=2, zorder=3,
               label=f"Random placement, in the bin ({len(ok)})")
    ax.scatter(*xy(bad), s=70, facecolors=SURFACE, edgecolors=COLOR["rtc"], linewidths=2, zorder=3,
               label=f"Random placement, failed ({len(bad)})")
    notes = {"16": "16 (box standing)"}
    for r in bad:
        ax.annotate(notes.get(r["id"], r["id"]), (float(r["azimuth_deg"]), float(r["orientation_deg"])),
                    textcoords="offset points", xytext=(8, -3), color=TEXT, fontsize=9)
    ax.set_xlabel("Base angle at the first grasp (°): where the box was")
    ax.set_ylabel("Gripper angle on the table (°)")
    ax.set_ylim(-5, 185)
    ax.set_yticks([0, 45, 90, 135, 180])
    ax.set_title("Where π0.5 grasped on random placements, against the training demos")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3, fontsize=9, handletextpad=0.3)
    fig.savefig(OUT / "placement_coverage.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    success_vs_time()
    control_timing()
    placement_coverage()
    for f in ("success_vs_time.png", "control_timing.png", "placement_coverage.png"):
        print(OUT / f)


if __name__ == "__main__":
    main()
