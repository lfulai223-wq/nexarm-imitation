#!/usr/bin/env python3
"""找试抓过程中伸进画面的人手（2026-11-22）。

操作者坐在支架相机画面的左下方，手伸进来一定会越过画面的左边或下边。每一帧缩成 160×90 灰度，
算最下面 8 行、最左边 8 列和这次开头 5 帧（中位数）的平均差；超过 12 的帧算"边上有东西"。
机械臂伸到右侧位置时也会越过下边，所以这只是筛选：--sheets 给可疑的试抓出几帧截图，人看是手还是机械臂，
确认的记进 results/operator_interventions.csv（eval_stages.py 和 export_results.py 读它）。
看不到不越过画面边缘的手。

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/find_hands.py act102_100k_r2 pi05_12k_rtc_r2
    $V scripts/find_hands.py --sheets outputs/hand_check act102_100k_r2 pi05_12k_rtc_r2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import failure_review as fr  # noqa: E402
from eval_stages import frame_times  # noqa: E402

W, H, EDGE = 160, 90, 8
HOT = 12.0  # 边缘平均差超过这个算"有东西"
SUSPECT = 8.0  # 整次最大值超过这个就出截图


def gray_frames(video: Path) -> np.ndarray:
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"scale={W}:{H}", "-f", "rawvideo",
                        "-pix_fmt", "gray", "-"], capture_output=True, check=True)
    return np.frombuffer(p.stdout, np.uint8).reshape(-1, H, W).astype(np.float32)


def edge_change(t: dict) -> np.ndarray:
    """每一帧左边、下边两条里变化较大的那个。"""
    f = gray_frames(fr.video(t, "front"))
    ref = np.median(f[:5], axis=0)
    bottom = np.abs(f[:, -EDGE:, :] - ref[-EDGE:, :]).mean(axis=(1, 2))
    left = np.abs(f[:, :, :EDGE] - ref[:, :EDGE]).mean(axis=(1, 2))
    return np.maximum(bottom, left)


def spans(times: np.ndarray, hot: np.ndarray, gap: int = 8) -> list[tuple[float, float]]:
    out: list[list[int]] = []
    for i in np.flatnonzero(hot):
        if out and i - out[-1][1] <= gap:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(float(times[a]), float(times[b])) for a, b in out]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="+", help="outputs/eval/ 下的评测名")
    ap.add_argument("--sheets", type=Path, metavar="DIR", help="给可疑的试抓出截图，存到 DIR")
    args = ap.parse_args()
    rows = []
    for name in args.sessions:
        for t in fr.load(name):
            e = edge_change(t)
            n = min(len(e), t["n_frames"])
            e = e[:n]
            times = frame_times(t["recording"], t["n_frames"])[:n]
            hot = spans(times, e > HOT)
            print(f"{name} 第 {t['trial']:>2} 次 {t['result']:<7} 边缘最大变化 {e.max():5.1f}  "
                  + ("，".join(f"{a:.1f}–{b:.1f} 秒" for a, b in hot) or "-"), flush=True)
            if args.sheets and e.max() > SUSPECT:
                idx: list[int] = []
                for i in np.argsort(e)[::-1]:  # 变化最大的几帧，彼此隔开 10 帧以上
                    if e[i] <= SUSPECT or len(idx) == 5:
                        break
                    if all(abs(i - j) > 10 for j in idx):
                        idx.append(int(i))
                idx.sort()
                g = fr.grab(fr.video(t, "front"), idx, (288, 162))
                rows.append((f"{name} t{t['trial']:02d} {t['result']}  frames at "
                             + ", ".join(f"{times[i]:.1f}s" for i in idx), [g[i] for i in idx]))
    if args.sheets and rows:
        args.sheets.mkdir(parents=True, exist_ok=True)
        for k in range(0, len(rows), 8):
            path = args.sheets / f"hands_{k // 8 + 1:02d}.jpg"
            fr.save_sheet(rows[k:k + 8], path)
            print(path)


if __name__ == "__main__":
    main()
