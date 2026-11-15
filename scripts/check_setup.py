#!/usr/bin/env python3
"""开工检查：每次录数据或评测之前跑一遍，确认硬件和桌面布置跟以前一样。

只读，不写舵机位置、不改力矩：

1. 设备都在不在（两条臂的串口、两个相机）；
2. 两个相机按录制设置能不能出帧、实际帧率和亮度（手臂相机帧率低说明光线不够）；
3. 支架相机里的 ArUco 码：0–3 号是否都认得出，和存下的基准位置比有没有偏（偏了说明相机被碰过）；
4. 从臂读舵机是否正常（2026-11-15 出现过 4–6 号关节大量读错，原因是接线/供电）；
5. 主臂和从臂的姿势差多少（差太多的话，录制程序一上力矩从臂会快速跳到主臂的姿势）。

在商家 LeRobot 的虚拟环境里运行（需要已经 `pip install -e ~/nexarm-teleop-latency`）：

    python scripts/check_setup.py                    # 全部检查
    python scripts/check_setup.py --save-reference   # 码贴好、相机定好之后存一次基准
    python scripts/check_setup.py --skip-arms        # 只查相机和码，不碰机械臂

注意：检查前要关掉 ffplay 预览，否则相机被占用打不开。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from nexarm_teleop import FOLLOWER_PORT, LEADER_PORT
from nexarm_teleop.driver import NexArmPort
from nexarm_teleop.protocol import map_leader_to_follower

ROOT = Path(__file__).resolve().parent.parent
REF_DIR = ROOT / "data" / "reference"
REF_JSON = REF_DIR / "aruco_front.json"
REF_IMG = REF_DIR / "front.jpg"

CAMERAS = {  # 与 configs/record_pilot.yaml 一致
    "front": dict(path="/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0",
                  w=640, h=360, fourcc="MJPG"),
    "wrist": dict(path="/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0",
                  w=640, h=480, fourcc="YUYV"),
}
RECORD_FPS = 15  # 与 configs/record_pilot.yaml 的 dataset.fps 一致
MARKER_IDS = [0, 1, 2, 3]
MOVED_PX = 3.0  # 码中心偏移超过这么多像素就提示相机可能动过
START_DIFF_WARN = 300  # 主从姿势差（计数）超过这个就提醒先对齐

results: list[tuple[str, str, str]] = []  # (级别, 项目, 说明)


def report(level: str, item: str, msg: str) -> None:
    results.append((level, item, msg))
    print(f"[{level}] {item}: {msg}", flush=True)


def check_devices() -> None:
    for name, path in [("主臂串口", LEADER_PORT), ("从臂串口", FOLLOWER_PORT),
                       ("支架相机", CAMERAS["front"]["path"]), ("手臂相机", CAMERAS["wrist"]["path"])]:
        if Path(path).exists():
            report("OK", name, "在")
        else:
            report("FAIL", name, f"找不到 {path}（没插、插错口，或者掉线了）")


def grab(cam: dict, secs: float = 3.0, keep: int = 10):
    """按录制设置取流 secs 秒，返回 (最后 keep 帧, 实际帧率, 平均亮度)。"""
    cap = cv2.VideoCapture(cam["path"], cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, 0.0, 0.0
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["w"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["h"])
    for _ in range(30):  # 让自动曝光稳定
        cap.read()
    frames, n, means, t0 = [], 0, [], time.perf_counter()
    while time.perf_counter() - t0 < secs:
        ok, f = cap.read()
        if ok:
            n += 1
            means.append(float(f.mean()))
            frames = (frames + [f])[-keep:]
    cap.release()
    return (frames or None), n / secs, (float(np.mean(means)) if means else 0.0)


def check_cameras() -> list[np.ndarray] | None:
    front = None
    for name, cam in CAMERAS.items():
        frames, fps, bright = grab(cam)
        label = "支架相机" if name == "front" else "手臂相机"
        if frames is None:
            report("FAIL", label, "打不开或没有画面（ffplay 预览没关？）")
            continue
        h, w = frames[-1].shape[:2]
        if (w, h) != (cam["w"], cam["h"]):
            report("FAIL", label, f"分辨率是 {w}×{h}，不是 {cam['w']}×{cam['h']}")
        level = "OK" if fps >= RECORD_FPS * 1.2 else "WARN"  # 要比录制频率高一截，取帧时才很少拿到重复帧
        hint = "" if level == "OK" else f"，不够录制用的 {RECORD_FPS} 帧（要 {RECORD_FPS * 1.2:.0f} 以上）" + ("，光线不够" if name == "wrist" else "")
        report(level, label, f"{fps:.1f} 帧/秒，平均亮度 {bright:.0f}{hint}")
        if name == "front":
            front = frames
    return front


def detect_markers(frame: np.ndarray) -> dict[int, np.ndarray]:
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def check_markers(frames: list[np.ndarray], save: bool) -> None:
    # 远处的码在 640×360 下只有 20 多像素，单帧偶尔认不出；取多帧，每个码用认出来的那些帧的中位数
    hits: dict[int, list[np.ndarray]] = {}
    for f in frames:
        for i, c in detect_markers(f).items():
            hits.setdefault(i, []).append(c)
    found = {i: np.median(np.stack(cs), axis=0) for i, cs in hits.items()}
    front = frames[-1]
    missing = [i for i in MARKER_IDS if i not in found]
    sizes = {i: float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)])) for i, c in found.items()}
    desc = "，".join(f"ID {i} 边长 {sizes[i]:.0f} 像素（{len(hits[i])}/{len(frames)} 帧认出）"
                     for i in sorted(found) if i in MARKER_IDS) or "一个都没认出来"
    report("OK" if not missing else "WARN", "ArUco 码", desc + (f"；缺 {missing}" if missing else ""))

    if save:
        if missing:
            report("FAIL", "保存基准", f"缺 {missing}，没有保存。先让 4 个码都能认出来")
            return
        REF_DIR.mkdir(parents=True, exist_ok=True)
        REF_JSON.write_text(json.dumps({str(i): found[i].tolist() for i in MARKER_IDS}, indent=1))
        cv2.imwrite(str(REF_IMG), front)
        report("OK", "保存基准", f"已存到 {REF_JSON.relative_to(ROOT)} 和 {REF_IMG.relative_to(ROOT)}")
        return

    if not REF_JSON.exists():
        report("WARN", "相机位置", "还没有基准，码贴好后运行一次 --save-reference")
        return
    ref = {int(k): np.array(v) for k, v in json.loads(REF_JSON.read_text()).items()}
    shifts = {i: float(np.linalg.norm(found[i].mean(0) - ref[i].mean(0))) for i in ref if i in found}
    if not shifts:
        report("FAIL", "相机位置", "一个基准码都没认出来，没法比较")
        return
    worst = max(shifts.values())
    text = "，".join(f"ID {i} 偏 {s:.1f}" for i, s in sorted(shifts.items()))
    if worst <= MOVED_PX:
        report("OK", "相机位置", f"和基准一致（{text} 像素）")
    else:
        report("WARN", "相机位置", f"{text} 像素：相机或码可能被碰过，对照 {REF_IMG.relative_to(ROOT)} 调回去")


def check_arms(secs: float = 10.0) -> None:
    # 从臂：桥接模式下以 50 Hz 只读位置，统计读错
    follower = None
    try:
        with NexArmPort(FOLLOWER_PORT) as arm:
            arm.enter_bridge_mode()
            n = bad = timeouts = 0
            t_end = time.perf_counter() + secs
            while time.perf_counter() < t_end:
                t = time.perf_counter()
                try:
                    p = arm.read_positions(timeout_s=0.1).positions
                except TimeoutError:
                    timeouts += 1
                    continue
                n += 1
                if all(0 <= x <= 4095 for x in p) and not (p[4] == 544 and p[5] == 8192):
                    follower = p
                else:
                    bad += 1
                time.sleep(max(0.0, 0.02 - (time.perf_counter() - t)))
        if n and bad == 0 and timeouts == 0:
            report("OK", "从臂读舵机", f"{n} 次读数全部正常")
        else:
            report("FAIL", "从臂读舵机",
                   f"{n} 次里读错 {bad} 次、超时 {timeouts} 次。先断电，检查 3 号关节到夹爪的舵机线和电源，再重开")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "从臂读舵机", f"{type(e).__name__}: {e}")

    # 主臂：读一次位置，换算到从臂坐标后和从臂当前姿势比
    try:
        with NexArmPort(LEADER_PORT) as arm:
            leader = arm.read_positions().positions
        report("OK", "主臂", "读位置正常")
    except Exception as e:  # noqa: BLE001
        report("FAIL", "主臂", f"{type(e).__name__}: {e}")
        return
    if follower is None:
        return
    target = map_leader_to_follower(leader)
    diffs = [abs(a - b) for a, b in zip(target[:5], follower[:5])]  # 夹爪开合不算
    worst = max(diffs)
    joint = diffs.index(worst) + 1
    if worst <= START_DIFF_WARN:
        report("OK", "主从姿势", f"最大相差 {worst} 计数（关节 {joint}）")
    else:
        report("WARN", "主从姿势",
               f"关节 {joint} 相差 {worst} 计数。开始录之前先把主臂摆成和从臂差不多的姿势，否则从臂会快速跳过去")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--save-reference", action="store_true", help="把当前 ArUco 码位置存为基准")
    ap.add_argument("--skip-arms", action="store_true", help="不检查机械臂")
    args = ap.parse_args()

    check_devices()
    front = check_cameras()
    if front is not None:
        check_markers(front, args.save_reference)
    if not args.skip_arms:
        check_arms()

    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print(f"\n结论：{len(fails)} 项不通过，{len(warns)} 项要注意。" + ("可以开始。" if not fails else "先处理不通过的项。"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
