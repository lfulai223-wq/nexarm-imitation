#!/usr/bin/env python3
"""失败复盘：每次失败归到一类原因，按模型和摆放位置汇总（2026-11-20）。

两部分数据：
1. 人工标注 results/failure_labels.csv，每次失败一行：场次、第几次、类别、是否夹在药盒一端（one_end）、把握、依据。
   标注时看每次失败的截图拼图（--sheets）：支架相机的开头一帧、第一次合上夹爪那一刻；手臂相机合上前一刻、合上后 0.67 秒。
   两张截图判断不了的，再看合上前后 -1 到 +3 秒的连续帧（--strip）。把握分 high / medium / low，low 的最好本人看录像复核。
2. 关节数据自动算（判定和 eval_stages.py 一样），都看第一次真正合上夹爪（连续合着 5 帧以上，滤掉门槛附近的抖动）：
   - 手腕朝向和横放 88°、竖放 0° 差多少（算法见 placement_coverage.py）。斜放每次摆的角度不一样，没有参照值，不算。
   - 夹紧了没有：合上后（跳过前 5 帧）夹爪指令减读数的中位数。夹爪被药盒顶住时读数停在药盒宽度，指令还要往里合，差值为正；
     合空或只是松松地夹着一端，读数跟着指令走，差值接近 0。第一轮成功夹起的 63 次里 53 次 >= 10 计数，
     54 次失败没有一次 >= 10，所以 10 可以当"夹紧了"的粗分界，但不能当成败的判定。

类别，按失败发生在哪一步排：
  wrong_place       去错地方：夹爪在离药盒较远的地方合上（包括在药盒上方的半空中合上），或一直没靠近药盒
  offset            位置偏一点：到了药盒上方，但偏到一侧或一端，一指压在药盒上、合在药盒边上
  orientation       朝向不对：夹爪方向和药盒差得多，包括在药盒上方斜着徘徊、一直不合爪
  aligned_no_grasp  看起来对准了：居中、方向也对，但一直不合爪，或合上后药盒没跟着起来（画面上看不出原因）
  pushed            合爪时把药盒推歪、推走
  lost_after_lift   夹起来以后失败：抬起或搬运中掉落、松在盒沿外，或时间到了还没松手
  unseen_condition  训练里没有的情况：旁边有别的东西、药盒立着

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/failure_review.py                                     # 汇总默认的几轮
    $V scripts/failure_review.py act102_100k_r2 pi05_12k_rtc_r2      # 汇总指定的几轮（没标注的失败会列出来）
    $V scripts/failure_review.py --sheets outputs/failure_review act102_100k_r2 pi05_12k_rtc_r2
    $V scripts/failure_review.py --strip outputs/failure_review act102_100k_r2:7 pi05_12k_rtc_r2:12
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_stages import CLOSE_DELTA, ZONE  # noqa: E402
from eval_trials import ROBUST8, SPOKEN, TRIALS, TRIALS2  # noqa: E402

LABELS = ROOT / "results" / "failure_labels.csv"
SESSIONS = ["act45_100k", "act45_100k_ab", "act102_100k_ab", "act102_100k_1118",
            "pi05_12k_smoke", "pi05_12k_rtc", "pi05_12k_rtc_free"]
CATEGORIES = {"wrong_place": "去错地方", "offset": "位置偏一点", "orientation": "朝向不对",
              "aligned_no_grasp": "对准了没夹住", "pushed": "推歪了药盒", "lost_after_lift": "夹起来后失败",
              "unseen_condition": "训练里没有的情况"}
CODE = {"wrong_place": "W", "offset": "O", "orientation": "R", "aligned_no_grasp": "A", "pushed": "P",
        "lost_after_lift": "L", "unseen_condition": "U"}
RIGHT_SIDE = {"右上", "右下", "中偏右", "右边中间", "中偏右上", "中偏右下"}  # 支架相机画面下沿那一片
NOMINAL = {"横放": 88.0, "竖放": 0.0}  # 训练数据里第一次合上夹爪时的朝向（placement_coverage.py）
DEG = 360 / 4096
MIN_CLOSE_FRAMES = 5
FIRM_GAP = 10  # 合上后指令减读数达到这么多计数，算被药盒顶住（夹紧了）
PLACEMENT_SETS = [("第一份位置表", TRIALS), ("第二份位置表", TRIALS2), ("鲁棒性 8 个位置", ROBUST8)]
ROBUST_TAG = "_rb_"  # 鲁棒性测试的评测名里带这个（如 act102_100k_rb_card），用 ROBUST8 位置表


def model_of(session: str) -> str:
    for prefix, name in (("act45", "ACT-45"), ("act102", "ACT-102"), ("pi05", "π0.5")):
        if session.startswith(prefix):
            return name
    return session


def placement_set(t: dict) -> str:
    if t["position"] == "随机":
        return "随机摆放"
    if ROBUST_TAG in t["session"]:
        return "鲁棒性 8 个位置"
    return next((name for name, lst in PLACEMENT_SETS[:2] if (t["position"], t["orientation"]) in lst), "其他")


def real_closes(states: np.ndarray) -> list[int]:
    """每次真正合上夹爪的帧号（连续合着 MIN_CLOSE_FRAMES 帧以上）。"""
    closed = states[:, 5] > states[:5, 5].min() + CLOSE_DELTA
    on = list(np.flatnonzero(np.diff(closed.astype(int)) == 1) + 1)
    if closed[0]:
        on.insert(0, 0)
    off = list(np.flatnonzero(np.diff(closed.astype(int)) == -1) + 1)
    return [i for i in on if next((k for k in off if k > i), len(states)) - i >= MIN_CLOSE_FRAMES]


def load(session: str) -> list[dict]:
    with (ROOT / "outputs" / "eval" / session / "results.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["result"] in ("success", "fail")]
    out = []
    for r in rows:
        rec = ROOT / r["recording"]
        files = sorted(rec.glob("data/chunk-*/file-*.parquet"))
        s = a = None
        if files:
            df = pd.concat([pd.read_parquet(p, columns=["observation.state", "action"]) for p in files])
            s, a = np.stack(df["observation.state"].to_numpy()), np.stack(df["action"].to_numpy())
        closes = real_closes(s) if s is not None else []
        dtheta = gap = None
        carried = False  # 第一次合上就把药盒带到了收纳盒上方
        if closes:
            c = closes[0]
            closed = s[:, 5] > s[:5, 5].min() + CLOSE_DELTA
            e = c + int(np.argmin(closed[c:])) if not closed[c:].all() else len(s)
            lo = c + MIN_CLOSE_FRAMES if e - c > MIN_CLOSE_FRAMES else c  # 跳过刚合上、还在动的几帧
            gap = float(np.median(a[lo:e, 5] - s[lo:e, 5]))
            carried = bool(((s[c:e, 0] >= ZONE[0]) & (s[c:e, 0] <= ZONE[1])).any())
            if r["orientation"] in NOMINAL:
                theta = ((s[c, 4] - s[c, 0]) % 2048) * DEG
                d = abs(theta - NOMINAL[r["orientation"]]) % 180
                dtheta = min(d, 180 - d)
        out.append(dict(session=session, trial=int(r["trial"]), position=r["position"], orientation=r["orientation"],
                        result=r["result"], recording=rec, n_frames=0 if s is None else len(s), closes=closes,
                        dtheta=dtheta, gap=gap, carried=carried))
    return out


def read_labels() -> dict[tuple[str, int], dict[str, str]]:
    with LABELS.open(newline="") as f:
        return {(r["session"], int(r["trial"])): r for r in csv.DictReader(f)}


def report(trials: list[dict], labels: dict[tuple[str, int], dict[str, str]]) -> None:
    fails = [t for t in trials if t["result"] == "fail"]
    missing = [t for t in fails if (t["session"], t["trial"]) not in labels]
    stale = sorted(k for k in labels if k[0] in {t["session"] for t in trials}
                   and not any((t["session"], t["trial"]) == k for t in fails))
    cat = {(t["session"], t["trial"]): labels.get((t["session"], t["trial"]), {}).get("category", "?") for t in fails}

    # 1. 各类失败的次数：按模型和摆法分组
    print("== 失败按原因分类（人工标注，results/failure_labels.csv）")
    groups: dict[str, Counter] = defaultdict(Counter)
    n_trials: Counter = Counter()
    one_end: Counter = Counter()
    for t in trials:
        g = f"{model_of(t['session'])}，{placement_set(t)}"
        n_trials[g] += 1
        if t["result"] == "fail":
            groups[g][cat[(t["session"], t["trial"])]] += 1
            one_end[g] += labels.get((t["session"], t["trial"]), {}).get("one_end") == "1"
    for g in n_trials:
        c = groups[g]
        parts = [f"{CATEGORIES.get(k, '未标注')} {c[k]}" for k in list(CATEGORIES) + ["?"] if c[k]]
        print(f"  {g}：{n_trials[g]} 次里失败 {sum(c.values())}" + (f"（{'，'.join(parts)}）" if parts else "")
              + (f"；其中夹在药盒一端的 {one_end[g]} 次" if one_end[g] else ""))

    # 2. 固定位置 × 场次：哪些位置每场都失败
    for set_name, lst in PLACEMENT_SETS:
        sessions = [s for s in dict.fromkeys(t["session"] for t in trials)
                    if any(placement_set(t) == set_name for t in trials if t["session"] == s)]
        if not sessions:
            continue
        by = {(t["session"], t["trial"]): t for t in trials}
        print(f"\n== {set_name}：每个位置在各场的结果（✓ 成功；失败写类别代号："
              + "，".join(f"{v} {CATEGORIES[k]}" for k, v in CODE.items()) + "；* 右侧，在支架相机画面下沿）")
        print("  " + " " * 14 + " ".join(f"{s[:16]:>16}" for s in sessions))
        for i, (pos, ori) in enumerate(lst, 1):
            cells = []
            for s in sessions:
                t = by.get((s, i))
                cells.append("" if t is None else "✓" if t["result"] == "success" else CODE.get(cat[(s, i)], "?"))
            mark = "*" if pos in RIGHT_SIDE else " "
            print(f"  {i:>2} {mark}{pos + ori:<10}" + " ".join(f"{c:>16}" for c in cells))

    # 3. 去错地方是不是集中在右侧
    fixed = [t for t in trials if placement_set(t) not in ("随机摆放", "其他")]
    wrong = [t for t in fixed if t["result"] == "fail" and cat[(t["session"], t["trial"])] == "wrong_place"]
    if fixed:
        right_share = sum(t["position"] in RIGHT_SIDE for t in fixed) / len(fixed)
        print(f"\n== 去错地方的位置：固定位置上 {len(wrong)} 次，其中右侧 {sum(t['position'] in RIGHT_SIDE for t in wrong)} 次"
              f"（右侧位置占全部固定试次的 {right_share:.0%}）")

    # 4. 第一次合上时的手腕朝向
    print("\n== 第一次真正合上夹爪时，手腕朝向和横放 88°、竖放 0° 差多少（度；只算横放、竖放，没合上的不算）")
    for m in dict.fromkeys(model_of(t["session"]) for t in trials):
        for ori in NOMINAL:
            for res in ("success", "fail"):
                v = [t for t in fixed if model_of(t["session"]) == m and t["orientation"] == ori and t["result"] == res]
                d = sorted(round(t["dtheta"]) for t in v if t["dtheta"] is not None)
                if v:
                    print(f"  {m:<7} {ori} {'成功' if res == 'success' else '失败'} {len(v):>2} 次，没合上 "
                          f"{sum(not t['closes'] for t in v)}；差 {d}" + (f"，超过 25° 的 {sum(x > 25 for x in d)} 次" if d else ""))
    ok = [t["dtheta"] for t in fixed if t["result"] == "success" and t["dtheta"] is not None]
    if ok:
        print(f"  成功的 {len(ok)} 次：中位数 {np.median(ok):.0f}°，90% 分位 {np.percentile(ok, 90):.0f}°")

    # 5. 第一次合上后夹紧了没有
    print(f"\n== 第一次合上后夹紧了没有（指令减读数，计数；>= {FIRM_GAP} 算被药盒顶住）")
    held = [t["gap"] for t in trials if t["result"] == "success" and t["carried"]]
    if held:
        print(f"  成功、而且第一次合上就带到收纳盒的 {len(held)} 次：中位数 {np.median(held):.0f}，"
              f">= {FIRM_GAP} 的 {sum(g >= FIRM_GAP for g in held)} 次")
    by_cat: dict[str, list[int]] = defaultdict(list)
    for t in fails:
        if t["gap"] is not None:
            by_cat[cat[(t["session"], t["trial"])]].append(round(t["gap"]))
    for k, v in by_cat.items():
        print(f"  失败，{CATEGORIES.get(k, '未标注')}：{sorted(v)}")

    if missing:
        print("\n还没标注的失败：" + "，".join(f"{t['session']} 第 {t['trial']} 次" for t in missing))
    if stale:
        print("\n标注表里有、但现在不是失败的：" + "，".join(f"{s} 第 {i} 次" for s, i in stale))


# ---- 截图拼图 ----

def grab(video: Path, idx: list[int], size: tuple[int, int]):
    from PIL import Image
    idx = sorted(set(idx))
    with tempfile.TemporaryDirectory() as td:
        sel = "+".join(f"eq(n\\,{i})" for i in idx)
        subprocess.run(["ffmpeg", "-v", "error", "-i", str(video), "-vf", f"select='{sel}'", "-vsync", "0",
                        f"{td}/f%03d.png"], check=True)
        ims = [Image.open(p).convert("RGB").resize(size) for p in sorted(Path(td).glob("f*.png"))]
    return dict(zip(idx, ims))


def video(t: dict, cam: str) -> Path:
    return t["recording"] / f"videos/observation.images.{cam}/chunk-000/file-000.mp4"


def label_of(t: dict) -> str:
    where = "random" if t["position"] == "随机" else f"{SPOKEN.get(t['position'], '?')} / {SPOKEN.get(t['orientation'], '?')}"
    return f"{t['session']} t{t['trial']:02d}  {where}  {t['result']}"


def save_sheet(rows: list, path: Path) -> None:
    from PIL import Image, ImageDraw
    width = max(sum(im.width for im in ims) for _, ims in rows)
    height = max(im.height for _, ims in rows for im in ims) + 18
    sheet = Image.new("RGB", (width, height * len(rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for j, (text, ims) in enumerate(rows):
        draw.text((4, j * height + 3), text, fill="black")
        x = 0
        for im in ims:
            sheet.paste(im, (x, j * height + 18))
            x += im.width
    sheet.save(path, quality=85)


def sheets(out: Path, trials: list[dict], which: str) -> None:
    """每次一行：支架相机开头、支架相机第一次合上时、手臂相机合上前 0.2 秒、合上后 0.67 秒。
    一直没合上的：支架相机开头和 1/3 处，手臂相机 1/4 和 1/2 处。"""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in (x for x in trials if x["result"] == which):
        n = t["n_frames"]
        if t["closes"]:
            c = t["closes"][0]
            fi, wi = [0, c], [max(0, c - 3), min(n - 1, c + 10)]
            tag = f"first close {c / 15:.1f}s, {len(t['closes'])} closes"
        else:
            fi, wi = [0, n // 3], [n // 4, n // 2]
            tag = "never closed (wrist at 1/4 and 1/2)"
        f = grab(video(t, "front"), fi, (384, 216))
        w = grab(video(t, "wrist"), wi, (288, 216))
        rows.append((f"{label_of(t)}  {tag}", [f[fi[0]], f[fi[1]], w[wi[0]], w[wi[1]]]))
    for k in range(0, len(rows), 5):
        path = out / f"{which}_{k // 5 + 1:02d}.jpg"
        save_sheet(rows[k:k + 5], path)
        print(path)


def strips(out: Path, trials: list[dict]) -> None:
    """第一次合上前后：手臂相机 -1、0、+0.5、+1、+2、+3 秒，加支架相机 +2 秒。"""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for t in trials:
        if not t["closes"]:
            print(f"{t['session']} 第 {t['trial']} 次夹爪没合上过，跳过")
            continue
        n, c = t["n_frames"], t["closes"][0]
        wi = [min(n - 1, max(0, c + round(o * 15))) for o in (-1, 0, 0.5, 1, 2, 3)]
        w = grab(video(t, "wrist"), wi, (192, 144))
        f = grab(video(t, "front"), [min(n - 1, c + 30)], (256, 144))
        rows.append((f"{label_of(t)}  first close {c / 15:.1f}s; wrist -1,0,+0.5,+1,+2,+3 s | front +2 s",
                     [w[i] for i in wi] + list(f.values())))
    for k in range(0, len(rows), 8):
        path = out / f"strips_{k // 8 + 1:02d}.jpg"
        save_sheet(rows[k:k + 8], path)
        print(path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sessions", nargs="*", help="outputs/eval/ 下的评测名；不给就用第一轮的 7 场")
    ap.add_argument("--sheets", type=Path, metavar="DIR", help="给这几场的失败出截图拼图，存到 DIR")
    ap.add_argument("--success", action="store_true", help="和 --sheets 一起用：给成功的出拼图（对照用）")
    ap.add_argument("--strip", type=Path, metavar="DIR",
                    help="给指定的几次出合上前后的连续帧，存到 DIR；这时 sessions 写成 名字:第几次")
    args = ap.parse_args()

    if args.strip:
        picks = [s.rsplit(":", 1) for s in args.sessions]
        if not picks or any(len(p) != 2 for p in picks):
            ap.error("--strip 时要写成 评测名:第几次，例如 act102_100k_r2:7")
        cache: dict[str, list[dict]] = {}
        chosen = []
        for name, trial in picks:
            cache.setdefault(name, load(name))
            chosen += [t for t in cache[name] if t["trial"] == int(trial)]
        strips(args.strip, chosen)
        return
    trials = [t for s in (args.sessions or SESSIONS) for t in load(s)]
    if args.sheets:
        sheets(args.sheets, trials, "success" if args.success else "fail")
        return
    report(trials, read_labels())


if __name__ == "__main__":
    main()
