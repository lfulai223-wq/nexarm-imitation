#!/usr/bin/env python3
"""把评测结果导出成 results/ 下的几张小表（进 git），README、报告和图都从这里取数。

原始录像和数据集（data/、outputs/）画面里有房间，不公开；这几张表只有数字，公开后别人也能核对和重画图。

  results/trials.csv           每次有效试抓一行：哪一轮、哪个模型、摆放、操作者判定，以及关节数据算出的
                               夹爪第一次合上、在收纳盒上方松手的时刻（和 eval_stages.py 同一个判定）、帧间隔统计；
                               有人手动过药盒的（results/operator_interventions.csv）记下人手进画面的时刻，只算那之前的松手
  results/frame_intervals.csv  固定第 6 个位置上 ACT-102、π0.5 同步、π0.5 RTC 各一次的逐帧间隔（看停顿）
  results/grasp_poses.csv      训练示范和随机摆放每次第一次合上夹爪时的位置、朝向，以及是否在训练覆盖范围内
                               （和 placement_coverage.py 同一个算法）

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/export_results.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_stages import CLOSE_DELTA, ZONE, frame_times, hand_times  # noqa: E402
from placement_coverage import NOVEL_PCT, THETA_SCALE, diffs, train_poses, trial_poses  # noqa: E402

OUT = ROOT / "results"

# (评测名, 模型, 日期, 每次多少秒, 摆法)
SESSIONS = [
    ("act45_100k", "ACT, 45 demos", "2026-11-16", 25, "fixed"),
    ("act45_100k_ab", "ACT, 45 demos", "2026-11-17", 40, "fixed"),
    ("act102_100k_ab", "ACT, 102 demos", "2026-11-17", 40, "fixed"),
    ("pi05_12k_smoke", "pi0.5, synchronous", "2026-11-18", 40, "fixed"),
    ("act102_100k_1118", "ACT, 102 demos", "2026-11-18", 40, "fixed"),
    ("pi05_12k_rtc", "pi0.5, RTC", "2026-11-18", 40, "fixed"),
    ("pi05_12k_rtc_free", "pi0.5, RTC", "2026-11-18", 40, "random"),
    ("act102_100k_r2", "ACT, 102 demos", "2026-11-22", 40, "fixed2"),
    ("pi05_12k_rtc_r2", "pi0.5, RTC", "2026-11-22", 40, "fixed2"),
]
PLACE = {"左上": "top-left", "右上": "top-right", "右下": "bottom-right", "左下": "bottom-left", "正中": "center",
         "中偏左": "center-left", "中偏右": "center-right", "随机": "random",
         "上边中间": "top edge middle", "右边中间": "right edge middle", "下边中间": "bottom edge middle",
         "左边中间": "left edge middle", "中偏左上": "center toward top-left", "中偏右上": "center toward top-right",
         "中偏右下": "center toward bottom-right", "中偏左下": "center toward bottom-left"}
ORIENT = {"横放": "horizontal", "竖放": "vertical", "斜放": "diagonal", "往另一边斜": "diagonal-mirrored", "随机": "random"}
TIMING_TRIAL = 6  # frame_intervals.csv 用哪个固定位置
TIMING_SESSIONS = ("act102_100k_1118", "pi05_12k_smoke", "pi05_12k_rtc")


def scored_rows(name: str) -> list[dict[str, str]]:
    """有效的试抓（成功或失败）；同一个位置重做过的取最后一次。"""
    with (ROOT / "outputs" / "eval" / name / "results.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["result"] in ("success", "fail")]
    last = {r["trial"]: r for r in rows}
    return sorted(last.values(), key=lambda r: int(r["trial"]))


def trial_stats(rec: Path, hand: float | None = None) -> dict:
    files = sorted(rec.glob("data/chunk-*/file-*.parquet"))
    s = np.stack(pd.concat([pd.read_parquet(p, columns=["observation.state"]) for p in files])["observation.state"].to_numpy())
    g, j1 = s[:, 5], s[:, 0]
    t = frame_times(rec, len(s))
    d = np.diff((g > g[:5].min() + CLOSE_DELTA).astype(int))
    on, off = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    rel = [i for i in off if ZONE[0] <= j1[i] <= ZONE[1] and (hand is None or t[i] < hand)]
    dt = np.diff(t) * 1000
    return {
        "first_close_s": f"{t[on[0]]:.2f}" if len(on) else "",
        "gripper_closes": len(on),
        "release_over_bin_s": f"{t[rel[0]]:.2f}" if rel else "",
        "hand_in_frame_s": "" if hand is None else f"{hand:.1f}",
        "frames": len(s),
        "median_frame_interval_ms": f"{np.median(dt):.1f}",
        "max_frame_interval_ms": f"{dt.max():.0f}",
        "frame_intervals_over_150ms": int((dt > 150).sum()),
        "timing_source": "frame_times" if (rec / "frame_times.txt").exists() else "frame_index/15",
        "_t": t,
    }


def write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{path.relative_to(ROOT)}  {len(rows)} 行")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    trials, intervals = [], []
    hands = hand_times()
    for name, policy, date, limit, mode in SESSIONS:
        for r in scored_rows(name):
            st = trial_stats(ROOT / r["recording"], hands.get((name, int(r["trial"]))))
            t = st.pop("_t")
            trials.append({"session": name, "policy": policy, "date": date, "time_limit_s": limit, "placements": mode,
                           "trial": int(r["trial"]), "position": PLACE[r["position"]], "orientation": ORIENT[r["orientation"]],
                           "operator_result": r["result"], **st})
            if name in TIMING_SESSIONS and int(r["trial"]) == TIMING_TRIAL:
                intervals += [{"session": name, "policy": policy, "trial": TIMING_TRIAL, "t_s": f"{a:.3f}",
                               "interval_ms": f"{(b - a) * 1000:.1f}"} for a, b in zip(t[:-1], t[1:])]
    write(OUT / "trials.csv", trials)
    write(OUT / "frame_intervals.csv", intervals)

    # 夹取位置：和 placement_coverage.py 同样的尺度和界限
    tp, eps = train_poses()
    scale = np.array([tp[:, 0].std(), tp[:, 1].std(), THETA_SCALE])

    def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sqrt(((diffs(a, b) / scale) ** 2).sum(-1))

    cut = np.percentile([np.delete(dist(tp[i], tp), i).min() for i in range(len(tp))], NOVEL_PCT)
    poses = [{"source": "training demo", "id": ep, "operator_result": "", "azimuth_deg": f"{p[0]:.1f}",
              "elbow_deg": f"{p[1]:.1f}", "orientation_deg": f"{p[2]:.1f}", "nearest_demo_distance": "",
              "within_coverage": ""} for p, ep in zip(tp, eps)]
    for r, p in trial_poses("pi05_12k_rtc_free"):
        if p is None:
            continue
        d = dist(p, tp).min()
        poses.append({"source": "random placement", "id": int(r["trial"]), "operator_result": r["result"],
                      "azimuth_deg": f"{p[0]:.1f}", "elbow_deg": f"{p[1]:.1f}", "orientation_deg": f"{p[2]:.1f}",
                      "nearest_demo_distance": f"{d:.2f}", "within_coverage": str(bool(d <= cut)).lower()})
    write(OUT / "grasp_poses.csv", poses)
    print(f"覆盖范围界限（训练示范留一法最近距离的 {NOVEL_PCT}% 分位）：{cut:.2f}")


if __name__ == "__main__":
    main()
