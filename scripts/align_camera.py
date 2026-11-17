#!/usr/bin/env python3
"""支架相机被碰歪以后，用语音报偏差，手动把它调回 ArUco 基准位置。

按 → 开始；之后每 2–3 秒语音报一个数：码中心和基准相差的最大像素数（按录制画面 640×360 算）。
轻轻转或推相机，数字变小就是方向对了。连续几次都不超过 2 像素会说 "Aligned"，然后退出。
Esc 中途退出。码被手挡住时会说 "Markers hidden"。

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/align_camera.py

只读相机，不碰机械臂。调好以后再跑一次 scripts/check_setup.py 确认。
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_setup as C  # noqa: E402

GOOD_PX = 2.0      # 录制画面里的像素
GOOD_ROUNDS = 3    # 连续几次达标才算调好
TIMEOUT_S = 15 * 60


def say(text: str) -> None:
    print(f"[voice] {text}", flush=True)
    subprocess.run(["spd-say", "--wait", text], check=False)


def measure(cap, ref: dict[int, np.ndarray], n: int = 4):
    hits: dict[int, list[np.ndarray]] = {}
    for _ in range(n):
        ok, f = cap.read()
        if not ok:
            continue
        for i, c in C.detect_markers(f).items():
            hits.setdefault(i, []).append(c)
    now = {i: np.median(np.stack(v), axis=0) for i, v in hits.items() if i in ref}
    if len(now) < len(ref):
        return None
    k = 640 / C.MARKER_W
    shifts = {i: (now[i].mean(0) - ref[i].mean(0)) * k for i in ref}
    src = np.float32([ref[i].mean(0) for i in sorted(ref)])
    dst = np.float32([now[i].mean(0) for i in sorted(ref)])
    m, _ = cv2.estimateAffinePartial2D(src, dst)
    angle = float(np.degrees(np.arctan2(m[1, 0], m[0, 0])))
    return max(float(np.linalg.norm(s)) for s in shifts.values()), shifts, angle


def main() -> None:
    ref = {int(k): np.array(v) for k, v in json.loads(C.REF_JSON.read_text()).items()}
    from pynput import keyboard

    start, stop = threading.Event(), threading.Event()

    def on_press(key):
        if key == keyboard.Key.right:
            start.set()
        elif key == keyboard.Key.esc:
            stop.set()

    keyboard.Listener(on_press=on_press).start()
    say("The stand camera has moved. Press right arrow when you are ready to adjust it")
    while not (start.is_set() or stop.is_set()):
        time.sleep(0.1)
    if stop.is_set():
        return

    cam = C.CAMERAS["front"]
    cap = cv2.VideoCapture(cam["path"], cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, C.MARKER_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, C.MARKER_H)
    for _ in range(10):
        cap.read()
    say("Adjust slowly. Smaller numbers are better")
    good, t0 = 0, time.monotonic()
    try:
        while not stop.is_set() and time.monotonic() - t0 < TIMEOUT_S:
            for _ in range(5):  # 丢掉说话期间积压的旧帧
                cap.grab()
            r = measure(cap, ref)
            if r is None:
                good = 0
                say("Markers hidden")
                continue
            worst, shifts, angle = r
            print("  " + "  ".join(f"ID {i}: dx={s[0]:+.1f} dy={s[1]:+.1f}" for i, s in sorted(shifts.items()))
                  + f"  旋转 {angle:+.2f}°  最大偏差 {worst:.1f}", flush=True)
            if worst <= GOOD_PX:
                good += 1
                if good >= GOOD_ROUNDS:
                    say("Aligned. Please let go of the camera")
                    print(f"调好了：最大偏差 {worst:.1f} 像素", flush=True)
                    return
                say("Good. Hold still")
            else:
                good = 0
                say(str(round(worst)))
    finally:
        cap.release()
    if not stop.is_set():
        say("Time is up. Tell Claude")


if __name__ == "__main__":
    main()
