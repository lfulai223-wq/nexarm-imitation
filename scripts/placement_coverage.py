#!/usr/bin/env python3
"""评测时的摆放和训练数据比一比：这些位置、角度训练时见过没有（2026-11-18 为随机摆放加的）。

为什么用关节读数，不用画面：支架相机在起始姿势时，药盒常被从臂挡住一大半；手臂相机视野窄，起始姿势时多半看不到药盒。
所以取每次第一次合上夹爪那一刻（夹爪读数比开头大 250 以上，和 eval_stages.py 一样）的关节读数：
    底座 j1 → 药盒在哪个方向（方位）；
    肘 j3 → 远近；
    朝向 θ = (j5 − j1) mod 2048 → 药盒怎么放的。手腕朝下时，手腕转角减去底座转角就是夹爪在桌面上的朝向；
        夹爪是对称的，转 180° 一样，所以对 2048 取模。训练数据里横放约 1000（88°）、竖放约 0。
都换成度。量的是"夹爪去夹的地方"：成功的次数就是药盒的位置；没夹住的次数是策略以为药盒在的地方，要对照录像看。
夹爪一次都没合上的，没有这个数。

每一次找训练数据里最近的一条：三个量各除以一个尺度（方位、远近用训练数据的标准差，朝向用 180° 均匀分布的标准差 52°）
后算欧氏距离。再算训练数据自己的留一法最近距离：一次摆放的最近距离比训练数据里 90% 的都大，就算训练时没见过的摆放。

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/placement_coverage.py pi05_12k_rtc_free
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "data" / "formal_clean_v2"
CLOSE_DELTA = 250  # 和 eval_stages.py 一样
DEG = 360 / 4096  # 舵机一圈 4096 个计数
THETA_SCALE = 180 / np.sqrt(12)  # 朝向在 0–180° 上均匀分布时的标准差，约 52°
NOVEL_PCT = 90  # 最近距离超过训练数据留一法最近距离的这个百分位，算没见过


def grasp_pose(states: np.ndarray) -> np.ndarray | None:
    """第一次合上夹爪那一刻的 (方位°, 肘°, 朝向°)；没合上过返回 None。"""
    closed = states[:, 5] > states[:5, 5].min() + CLOSE_DELTA  # 和 eval_stages.py 一样取开头最小
    if not closed.any():
        return None
    s = states[int(np.argmax(closed))]
    return np.array([s[0] * DEG, s[2] * DEG, ((s[4] - s[0]) % 2048) * DEG])


def diffs(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """逐项差的绝对值（度），朝向按 180° 周期取近的那一边。"""
    d = np.abs(a - b)
    d[..., 2] = np.minimum(d[..., 2], 180 - d[..., 2])
    return d


def train_poses() -> tuple[np.ndarray, list[int]]:
    files = sorted(TRAIN.glob("data/chunk-*/file-*.parquet"))
    df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state"]) for f in files)
    poses, eps = [], []
    for ep, g in df.groupby("episode_index"):
        p = grasp_pose(np.stack(g.sort_values("frame_index")["observation.state"].to_numpy()))
        if p is not None:
            poses.append(p)
            eps.append(int(ep))
    return np.array(poses), eps


def trial_poses(name: str) -> list[tuple[dict[str, str], np.ndarray | None]]:
    with (ROOT / "outputs" / "eval" / name / "results.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["result"] in ("success", "fail")]
    out = []
    for r in rows:
        files = sorted((ROOT / r["recording"]).glob("data/chunk-*/file-*.parquet"))
        p = None
        if files:
            df = pd.concat([pd.read_parquet(f, columns=["observation.state"]) for f in files])
            p = grasp_pose(np.stack(df["observation.state"].to_numpy()))
        out.append((r, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="outputs/eval/ 下的评测名")
    args = ap.parse_args()

    tp, eps = train_poses()
    scale = np.array([tp[:, 0].std(), tp[:, 1].std(), THETA_SCALE])

    def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.sqrt(((diffs(a, b) / scale) ** 2).sum(-1))

    loo = np.array([np.delete(dist(tp[i], tp), i).min() for i in range(len(tp))])
    cut = np.percentile(loo, NOVEL_PCT)
    print(f"训练数据：{len(tp)} 条有合上夹爪（共 102 条）。方位 {tp[:, 0].min():.0f}–{tp[:, 0].max():.0f}°，"
          f"肘 {tp[:, 1].min():.0f}–{tp[:, 1].max():.0f}°，尺度 方位 {scale[0]:.1f}° 肘 {scale[1]:.1f}° 朝向 {scale[2]:.0f}°")
    print(f"训练数据留一法最近距离：中位数 {np.median(loo):.2f}，{NOVEL_PCT}% 分位 {cut:.2f}（超过它算没见过）\n")

    rows = trial_poses(args.name)
    novel = {"success": [0, 0], "fail": [0, 0]}  # 结果 → [没见过的次数, 有合上夹爪的次数]
    print(f"== {args.name}")
    for r, p in rows:
        head = f"{r['trial']:>2} {r['position']}{r['orientation']} {r['result']:<7}"
        if p is None:
            print(f"{head} 夹爪没合上，没有夹取位置")
            continue
        d = dist(p, tp)
        j = int(d.argmin())
        dd = diffs(p, tp[j])
        flag = d[j] > cut
        novel[r["result"]][0] += flag
        novel[r["result"]][1] += 1
        print(f"{head} 方位 {p[0]:6.1f}° 肘 {p[1]:6.1f}° 朝向 {p[2]:5.1f}°  最近的训练第 {eps[j]:>3} 条：距离 {d[j]:.2f}"
              f"（方位差 {dd[0]:.1f}° 肘差 {dd[1]:.1f}° 朝向差 {dd[2]:.0f}°）{'  ← 训练时没见过' if flag else ''}")
    for k, (n_new, n) in novel.items():
        print(f"{'成功' if k == 'success' else '失败'}且合上过夹爪 {n} 次，其中训练时没见过的 {n_new} 次")


if __name__ == "__main__":
    main()
