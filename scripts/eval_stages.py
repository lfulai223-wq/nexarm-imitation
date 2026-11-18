#!/usr/bin/env python3
"""评测结果按关节数据逐次分析：夹爪什么时候合上、什么时候在收纳盒上方松手（数据判定的成功）。

    $V scripts/eval_stages.py act45_100k                                  # 一轮评测
    $V scripts/eval_stages.py act45_100k_ab act102_100k_ab --cut 25       # 两个模型交替测，加逐个位置的配对比较

判定和 2026-11-16 在基线 20 次上验证过的一样（和操作者的判断 20 次全部一致）：
夹爪读数比开头张开时（前 5 帧最小值）大 250 以上算合上；合上→张开的那一刻底座关节在 1509–1692
（训练数据里松手时底座关节在 1559–1642，两头各放宽 50）算放进了收纳盒。
每次 40 秒的评测报两个数：--cut 秒内放进（数据判定）和整次成功（操作者判定）。
"""

from __future__ import annotations

import argparse
import csv
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FPS = 15
ZONE = (1509, 1692)  # 松手时底座关节（j1）的范围
CLOSE_DELTA = 250  # 夹爪读数比开头大这么多算合上


def frame_times(rec_dir: Path, n: int) -> np.ndarray:
    """每一帧的时间（秒，从第一帧算起）。有 rollout_safe.py 记下的 frame_times.txt（2026-11-18 起）就用它：
    控制循环超时的那一步不补帧，π0.5 每次重新预测要停约 0.25 秒，按"第几帧 / 15"算会偏短。
    以前的录像没有这个文件，按帧数算（ACT 每次只超时几步，差别不到 1%）。"""
    f = rec_dir / "frame_times.txt"
    if f.exists():
        t = np.loadtxt(f, ndmin=1)
        if len(t) == n:
            return t
        print(f"   {f} 有 {len(t)} 行，录像有 {n} 帧，对不上，改按帧数算时间")
    return np.arange(n) / FPS


def analyse(name: str, cut: float) -> dict[int, dict[str, bool]]:
    """打印每次的阶段，返回 {第几次: {"op": 操作者判成功, "data": cut 秒内数据判定放进}}。"""
    with (ROOT / "outputs" / "eval" / name / "results.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["result"] in ("success", "fail")]
    print(f"== {name}")
    out: dict[int, dict[str, bool]] = {}
    n_rel = 0
    for r in rows:
        d = ROOT / r["recording"]
        files = sorted(d.glob("data/chunk-*/file-*.parquet"))
        if not files:
            print(f"{r['trial']:>2} {r['position']}{r['orientation']}: 没有录像数据（{d}）")
            continue
        df = pd.concat([pd.read_parquet(p) for p in files])
        s = np.stack(df["observation.state"].to_numpy())
        g, j1 = s[:, 5], s[:, 0]
        t = frame_times(d, len(df))
        closed = g > g[:5].min() + CLOSE_DELTA  # 开头张开时的读数；取最小，药盒就在爪下、一开始就合上时也对（2026-11-18 随机第 17 次）
        on = np.flatnonzero(np.diff(closed.astype(int)) == 1) + 1
        off = np.flatnonzero(np.diff(closed.astype(int)) == -1) + 1
        rel = [i for i in off if ZONE[0] <= j1[i] <= ZONE[1]]
        t_rel = float(t[rel[0]]) if rel else None
        n_rel += t_rel is not None
        stage = ("放进收纳盒" if rel else
                 "夹住后没送到" if any(ZONE[0] <= x <= ZONE[1] for x in j1[closed]) else
                 (f"合上 {len(on)} 次没夹起来" if len(on) else "夹爪没合上"))
        print(f"{r['trial']:>2} {r['position']}{r['orientation']:<5} 操作者:{r['result']:<7} 帧 {len(df):>3}"
              f"（{t[-1]:.1f} 秒）  合上 {[round(float(t[i]), 1) for i in on]}  "
              f"松手 {None if t_rel is None else round(t_rel, 1)}  -> {stage}")
        out[int(r["trial"])] = {"op": r["result"] == "success", "data": t_rel is not None and t_rel <= cut}
    n_op = sum(r["result"] == "success" for r in rows)
    n_cut = sum(x["data"] for x in out.values())
    print(f"有效 {len(rows)} 次：操作者判成功 {n_op}，数据判定放进 {n_rel}（其中 {cut:g} 秒内 {n_cut}）\n")
    return out


def paired(a: dict[int, dict[str, bool]], b: dict[int, dict[str, bool]], na: str, nb: str, cut: float) -> None:
    """同一个摆放位置上两个模型的结果配对比较。只有一个成功的位置才有信息；
    用精确的配对符号检验（McNemar）算双侧 p 值，20 个位置时差距要很大才会显著。"""
    common = sorted(set(a) & set(b))
    print(f"== 配对比较（{na} 对 {nb}，{len(common)} 个位置两个都测了）")
    for key, label in (("data", f"{cut:g} 秒内（数据判定）"), ("op", "整次（操作者判定）")):
        both = sum(a[t][key] and b[t][key] for t in common)
        only_a = sum(a[t][key] and not b[t][key] for t in common)
        only_b = sum(b[t][key] and not a[t][key] for t in common)
        n = only_a + only_b
        p = min(1.0, 2 * sum(comb(n, k) for k in range(min(only_a, only_b) + 1)) / 2 ** n) if n else 1.0
        print(f"{label}：{na} 成功 {both + only_a}，{nb} 成功 {both + only_b}；两个都成功 {both}，只有 {na} {only_a}，"
              f"只有 {nb} {only_b}，都失败 {len(common) - both - n}；配对符号检验双侧 p = {p:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="+", help="outputs/eval/ 下的评测名字；给两个时加配对比较")
    ap.add_argument("--cut", type=float, default=25.0, help="数据判定只算这么多秒以内松手的")
    args = ap.parse_args()
    results = [analyse(n, args.cut) for n in args.names]
    if len(results) == 2:
        paired(results[0], results[1], args.names[0], args.names[1], args.cut)


if __name__ == "__main__":
    main()
