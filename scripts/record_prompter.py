#!/usr/bin/env python3
"""录制时用语音报下一条摆在哪里，不用对照表（2026-11-17 第二批起，本人要求：对着表容易摆错）。

跟着录制程序的日志走：每次听到 "Reset the environment" 后，报下一条的编号、位置和方向；
复位时按了 ← 作废刚录的那条，会重新报这一条。只读日志，不碰机械臂和录制程序。

    $V scripts/record_prompter.py outputs/logs/record_batch2_part2.log

摆放顺序在下面的 PLAN 里，和 docs/recording_guide.md 第五节的第二批表一致，
只是第一轮横放按实际录的情况调整过（见 PLAN 上面的注释）。
"""

from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

POS = {"左上": "top left", "右上": "top right", "右下": "bottom right", "左下": "bottom left",
       "正中": "center", "中偏左": "center left", "中偏右": "center right"}
ORI = {"横放": "horizontal", "横放稍歪": "horizontal, slightly tilted", "竖放": "vertical",
       "斜放": "diagonal", "往另一边斜": "other diagonal"}
# 每轮 11 个位置；True 表示同一区域的第二次，和第一次错开几厘米
ROW = [("左上", False), ("右上", False), ("右下", False), ("左下", False), ("正中", False), ("中偏左", False),
       ("中偏右", False), ("右上", True), ("中偏右", True), ("左下", True), ("中偏左", True)]

# 第二批第一轮（横放）实际录了 50–57：53 时间到了没松手、54 开头药盒已经夹在手里，两条都不能用；
# 56、57 摆成了别的方向（还能用，算作别的方向）。所以第一轮剩下的 3 个之后，把这 4 个横放补录一遍。
FIRST = [("中偏右", True, "横放"), ("左下", True, "横放"), ("中偏左", True, "横放"),
         ("左下", False, "横放"), ("正中", False, "横放"), ("中偏右", False, "横放"), ("右上", True, "横放")]
PLAN = {58 + i: x for i, x in enumerate(FIRST + [(p, s, o) for o in ("横放稍歪", "竖放", "斜放", "往另一边斜")
                                               for p, s in ROW])}
LAST = max(PLAN)  # 108

SHIFTED = "shifted a few centimeters"


def say(text: str) -> None:
    print(f"[voice] {text}", flush=True)
    subprocess.run(["spd-say", "--wait", text], check=False)


def describe(n: int) -> tuple[str, str]:
    p, s, o = PLAN[n]
    zh = f"{p}{'（错开几厘米）' if s else ''}，{o}"
    en = f"{POS[p]}{', ' + SHIFTED if s else ''}. {ORI[o]}"
    return zh, en


def announce(n: int, redo: bool = False) -> None:
    if n in PLAN:
        zh, en = describe(n)
        print(f"{'重录' if redo else '下一条'} episode {n}：{zh}", flush=True)
        say(f"{'Redo' if redo else 'Next,'} episode {n}. {en}.")
    elif n == LAST + 1:
        say("That was the last one. Put the arm in the rest pose, then press escape.")
    else:
        say(f"Episode {n} is not in the plan. Tell Claude.")


def follow(path: Path):
    while not path.exists():
        time.sleep(0.2)
    with path.open(errors="replace") as f:
        buf = ""
        while True:
            chunk = f.readline()
            if not chunk:
                time.sleep(0.2)
                continue
            buf += chunk
            if not buf.endswith("\n"):
                continue
            line, buf = buf, ""
            for part in line.split("\r"):  # 进度条用 \r 刷新，拆开
                yield part


def main() -> None:
    path = Path(sys.argv[1])
    cur, redo, phase, first = None, False, "rec", True
    for line in follow(path):
        m = re.search(r"Recording episode (\d+)", line)
        if m:
            cur, redo, phase = int(m.group(1)), False, "rec"
        elif "Left arrow key pressed" in line:
            redo = True
            if phase == "reset" and cur is not None:
                announce(cur, redo=True)
        elif "Reset the environment" in line and cur is not None:
            phase = "reset"
            time.sleep(1.5)  # 等录制程序自己说完 "Reset the environment"
            announce(cur if redo else cur + 1, redo=redo and not first)  # 开头那段按 ← 作废是规定动作，不算重录
            first = False
        elif "Escape key pressed" in line or "RECORD EXIT" in line or "Stop recording" in line:
            print("录制结束，提示程序退出", flush=True)
            return


if __name__ == "__main__":
    main()
