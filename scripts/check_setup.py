#!/usr/bin/env python3
"""开工检查：每次录数据或评测之前跑一遍，确认硬件和桌面布置跟以前一样。

只读，不写舵机位置、不改力矩（唯一会改的是相机的曝光设置）：

1. 设备都在不在（两条臂的串口、两个相机）；
2. 把两个相机设成固定曝光（相机拔插或断电后会恢复自动曝光，所以每次都要设），
   再按录制设置看能不能出帧、实际帧率和亮度；
3. 支架相机里的 ArUco 码（用 1920×1080 识别，远处的码才够大）：0–3 号是否都认得出，
   和存下的基准位置比有没有偏（偏了说明相机被碰过）；
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
REF_LIGHT = REF_DIR / "brightness.json"  # 存基准时两个相机的平均亮度，用来发现灯光变了
LIGHT_TOL = 0.15  # 亮度和基准差超过 15% 就提示

CAMERAS = {  # 分辨率和格式与录制配置一致；曝光是 2026-11-15 在顶灯下实测选的
    # 支架相机：曝光 30 ms 时亮度和自动曝光差不多（约 142），帧率不受曝光影响，固定是为了画面亮度不随手臂进出画面变化
    "front": dict(path="/dev/v4l/by-id/usb-icSpring_icspring_camera_202404160005-video-index0",
                  w=640, h=360, fourcc="MJPG", exposure=300, brightness=0),
    # 手臂相机：自动曝光时只有约 11 帧、白色药盒过曝；固定 25 ms 曝光 + 亮度补偿 64 时 22.8 帧、亮度约 88、不过曝
    "wrist": dict(path="/dev/v4l/by-id/usb-icSpring_icspring_camera-video-index0",
                  w=640, h=480, fourcc="YUYV", exposure=250, brightness=64),
}
MARKER_W, MARKER_H = 1920, 1080  # 识别码用的分辨率，和 640×360 是同一个视野
RECORD_FPS = 15  # 与 configs/record_pilot.yaml 的 dataset.fps 一致
MARKER_IDS = [0, 1, 2, 3]
# 存基准至少要认出几个码：3 个码的 12 个角点足够判断相机有没有动、换算桌面坐标。
# 2026-11-15 的布置里 0 号码在从臂休息姿势时被小臂挡住，基准只用 1–3 号。
MIN_REF_MARKERS = 3
MOVED_PX = 3.0 * MARKER_W / 640  # 码中心偏移超过录制画面里 3 像素就提示相机可能动过
START_DIFF_WARN = 300  # 主从姿势差（计数）超过这个就提醒先对齐

results: list[tuple[str, str, str]] = []  # (级别, 项目, 说明)
brightness: dict[str, float] = {}  # 这次测到的各相机平均亮度


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


def apply_exposure(cam: dict) -> bool:
    """设固定曝光。设置保存在相机里，关掉再打开（包括 LeRobot 打开）仍然有效，拔插或断电后恢复自动。"""
    cap = cv2.VideoCapture(cam["path"], cv2.CAP_V4L2)
    if not cap.isOpened():
        return False
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2：1 = 手动曝光
    cap.set(cv2.CAP_PROP_EXPOSURE, cam["exposure"])  # 单位 100 微秒
    cap.set(cv2.CAP_PROP_BRIGHTNESS, cam["brightness"])
    ok = cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) == 1 and cap.get(cv2.CAP_PROP_EXPOSURE) == cam["exposure"]
    cap.release()
    return ok


def grab(cam: dict, secs: float = 3.0, keep: int = 10, size: tuple[int, int] | None = None):
    """按录制设置（或给定分辨率）取流 secs 秒，返回 (最后 keep 帧, 实际帧率, 平均亮度)。"""
    cap = cv2.VideoCapture(cam["path"], cv2.CAP_V4L2)
    if not cap.isOpened():
        return None, 0.0, 0.0
    w, h = size or (cam["w"], cam["h"])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    for _ in range(20):
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


def check_cameras() -> None:
    for name, cam in CAMERAS.items():
        label = "支架相机" if name == "front" else "手臂相机"
        if not apply_exposure(cam):
            report("FAIL", label, "设不了固定曝光，或者打不开（ffplay 预览没关？）")
            continue
        frames, fps, bright = grab(cam)
        if frames is None:
            report("FAIL", label, "打不开或没有画面（ffplay 预览没关？）")
            continue
        h, w = frames[-1].shape[:2]
        if (w, h) != (cam["w"], cam["h"]):
            report("FAIL", label, f"分辨率是 {w}×{h}，不是 {cam['w']}×{cam['h']}")
        level = "OK" if fps >= RECORD_FPS * 1.2 else "WARN"  # 要比录制频率高一截，取帧时才很少拿到重复帧
        hint = "" if level == "OK" else f"，不够录制用的 {RECORD_FPS} 帧（要 {RECORD_FPS * 1.2:.0f} 以上）" + ("，光线不够" if name == "wrist" else "")
        report(level, label, f"固定曝光，{fps:.1f} 帧/秒，平均亮度 {bright:.0f}{hint}")
        brightness[name] = bright


def check_light(save: bool) -> None:
    """曝光是固定的，所以画面亮度只取决于灯光：和基准差太多说明灯光变了（窗帘没拉、换了灯、白天漏光）。"""
    if len(brightness) < 2:
        return
    if save:
        REF_DIR.mkdir(parents=True, exist_ok=True)
        REF_LIGHT.write_text(json.dumps(brightness, indent=1))
        return
    if not REF_LIGHT.exists():
        report("WARN", "灯光", "还没有基准亮度，运行一次 --save-reference")
        return
    ref = json.loads(REF_LIGHT.read_text())
    diffs = {k: (brightness[k] - ref[k]) / ref[k] for k in ref if k in brightness}
    text = "，".join(f"{'支架' if k == 'front' else '手臂'}相机 {brightness[k]:.0f}（基准 {ref[k]:.0f}）" for k in diffs)
    if all(abs(d) <= LIGHT_TOL for d in diffs.values()):
        report("OK", "灯光", f"和基准一致：{text}")
    else:
        report("WARN", "灯光", f"{text}，差超过 {LIGHT_TOL:.0%}：窗帘拉严了吗？开的是同一盏灯吗？")


def detect_markers(frame: np.ndarray) -> dict[int, np.ndarray]:
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def check_markers(save: bool) -> None:
    # 用 1920×1080 识别（远处的码在 640×360 下只有 20 多像素，认不稳）；取多帧，每个码用认出来的那些帧的中位数
    frames, _, _ = grab(CAMERAS["front"], secs=1.5, size=(MARKER_W, MARKER_H))
    if frames is None:
        report("FAIL", "ArUco 码", "支架相机打不开")
        return
    hits: dict[int, list[np.ndarray]] = {}
    for f in frames:
        for i, c in detect_markers(f).items():
            hits.setdefault(i, []).append(c)
    found = {i: np.median(np.stack(cs), axis=0) for i, cs in hits.items()}
    front = frames[-1]
    missing = [i for i in MARKER_IDS if i not in found]
    sizes = {i: float(np.mean([np.linalg.norm(c[k] - c[(k + 1) % 4]) for k in range(4)])) for i, c in found.items()}
    desc = "，".join(f"ID {i} 边长 {sizes[i] * 640 / MARKER_W:.0f} 像素（录制画面里；{len(hits[i])}/{len(frames)} 帧认出）"
                     for i in sorted(found) if i in MARKER_IDS) or "一个都没认出来"
    report("OK" if not missing else "WARN", "ArUco 码", desc + (f"；缺 {missing}" if missing else ""))

    if save:
        present = [i for i in MARKER_IDS if i in found]
        if len(present) < MIN_REF_MARKERS:
            report("FAIL", "保存基准", f"只认出 {present}，至少要 {MIN_REF_MARKERS} 个码，没有保存")
            return
        REF_DIR.mkdir(parents=True, exist_ok=True)
        REF_JSON.write_text(json.dumps({str(i): found[i].tolist() for i in present}, indent=1))
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
    text = "，".join(f"ID {i} 偏 {s * 640 / MARKER_W:.1f}" for i, s in sorted(shifts.items()))  # 换算到录制画面的像素
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
    check_cameras()
    check_light(args.save_reference)
    check_markers(args.save_reference)
    if not args.skip_arms:
        check_arms()

    fails = [r for r in results if r[0] == "FAIL"]
    warns = [r for r in results if r[0] == "WARN"]
    print(f"\n结论：{len(fails)} 项不通过，{len(warns)} 项要注意。" + ("可以开始。" if not fails else "先处理不通过的项。"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
