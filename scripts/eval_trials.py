#!/usr/bin/env python3
"""真机评测：按固定的 20 个摆放位置逐次试抓，全程用键盘操作，结果写进 CSV。

每次试抓调用一次 LeRobot 的 lerobot-rollout（sentry 模式，把画面录进数据集，方便回放），
跑 --duration 秒后从臂自动回到起始姿势。ACT 和 π0.5 用同一份位置表，保证对比公平。

按键（全局监听，先点一下桌面空白处，免得按键落到别的程序里）：
    →    开始这一次试抓（药盒按语音报的编号摆好、从臂在起始姿势时再按）
    s    这次成功（药盒进了收纳盒）
    f    这次失败
    r    这次不算，重做（比如药盒没摆好）
    t    开 / 关遥操作（用主臂把从臂带回起始姿势；开之前先把主臂摆成和从臂差不多的姿势）
    Esc  试抓进行中：急停（立即结束程序，从臂停在原地保持力矩）；空闲时：退出

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/eval_trials.py --name act45_100k \\
        --policy outputs/train/act_medbox_45ep/checkpoints/100000/pretrained_model
    $V scripts/eval_trials.py --selftest    # 不连硬件、不监听键盘，用脚本化的按键检查流程

开始前：跑过 scripts/check_setup.py；从臂停在起始姿势。位置表见 TRIALS 和 docs/recording_guide.md。
"""

from __future__ import annotations

import argparse
import csv
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TASK = "Pick up the medicine box and put it in the bin."

# 位置名和录数据时的表一致（以操作者坐的位置为准）。方向：横放 = 长边和面前的桌沿平行，竖放 = 长边指向对面
TRIALS = [
    ("左上", "横放"), ("右上", "横放"), ("右下", "横放"), ("左下", "横放"), ("正中", "横放"),
    ("左上", "竖放"), ("右上", "竖放"), ("右下", "竖放"), ("左下", "竖放"), ("正中", "竖放"),
    ("左上", "斜放"), ("右上", "斜放"), ("右下", "斜放"), ("左下", "斜放"), ("正中", "斜放"),
    ("中偏左", "横放"), ("中偏右", "横放"), ("中偏左", "竖放"), ("中偏右", "竖放"), ("正中", "往另一边斜"),
]


def say(text: str, quiet: bool = False) -> None:
    print(f"[voice] {text}", flush=True)
    if not quiet:
        subprocess.run(["spd-say", "--wait", text], check=False)


class Keys:
    """把全局按键变成事件队列；selftest 时直接往队列里塞事件。"""

    NAMES = {"right": "start", "s": "success", "f": "fail", "r": "redo", "t": "teleop", "esc": "esc"}

    def __init__(self, scripted: list[str] | None = None):
        self.q: queue.Queue[str] = queue.Queue()
        self.estop = threading.Event()  # Esc 一按下就置位，不受队列里排在前面的其他按键影响
        self.scripted = scripted is not None
        self.listener = None
        if scripted is not None:
            for e in scripted:
                self.q.put(e)
            return
        from pynput import keyboard

        def on_press(key):
            name = key.name if isinstance(key, keyboard.Key) else getattr(key, "char", None)
            if name in self.NAMES:
                if name == "esc":
                    self.estop.set()
                self.q.put(self.NAMES[name])

        self.listener = keyboard.Listener(on_press=on_press)
        self.listener.start()

    def get(self, allowed: set[str], timeout: float | None = None) -> str | None:
        end = None if timeout is None else time.monotonic() + timeout
        while True:
            left = None if end is None else max(0.0, end - time.monotonic())
            try:
                e = self.q.get(timeout=left)
            except queue.Empty:
                return None
            if e in allowed:
                return e

    def peek(self) -> str | None:
        with self.q.mutex:
            return self.q.queue[0] if self.q.queue else None

    def drain(self) -> None:
        while not self.q.empty():
            self.q.get_nowait()


def rollout_cmd(args, dataset_root: Path) -> list[str]:
    cmd = [PY, "-m", "lerobot.scripts.lerobot_rollout", "--config_path=configs/rollout_base.yaml",
           f"--policy.path={args.policy}", f"--duration={args.duration}", "--play_sounds=false",
           "--strategy.type=sentry", f"--dataset.repo_id=local/rollout_{args.name}", f"--dataset.root={dataset_root}",
           f"--dataset.single_task={TASK}", "--dataset.fps=15", "--dataset.push_to_hub=false", "--dataset.video=true"]
    if (dataset_root / "meta" / "info.json").exists():
        cmd.append("--resume=true")
    return cmd


def run_child(cmd: list[str], log_path: Path, keys: Keys, quiet: bool) -> str:
    """运行一个子进程（试抓或遥操作），返回 'done' 或 'estop'。试抓期间按 Esc 急停。"""
    with log_path.open("w") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        keys.estop.clear()
        while p.poll() is None:
            if keys.scripted and keys.peek() == "esc":  # selftest：脚本里的 Esc 当作试抓中的急停
                keys.get({"esc"})
                keys.estop.set()
            if keys.estop.is_set():
                p.send_signal(signal.SIGKILL)  # 不走收尾流程：不回位，从臂停在原地保持力矩
                p.wait()
                say("Emergency stop", quiet)
                return "estop"
            time.sleep(0.05)
    return "done"


def teleop(keys: Keys, log_dir: Path, quiet: bool, selftest: bool) -> None:
    # 遥操作一启动，从臂会快速转到主臂的姿势，所以按 T 之前主臂要先摆成和从臂差不多
    say("Teleop on. Press T again when the arm is at the start pose", quiet)
    cmd = [PY, "-c", "import time; time.sleep(0.2)"] if selftest else \
        [PY, "-m", "lerobot.scripts.lerobot_teleoperate", "--config_path=configs/teleop_practice.yaml"]
    with (log_dir / f"teleop_{datetime.now():%H%M%S}.log").open("w") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        keys.get({"teleop"})
        if p.poll() is None:
            p.send_signal(signal.SIGINT)  # 正常退出：从臂停在当前姿势、保持力矩
            p.wait(timeout=15)
    say("Teleop off", quiet)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=str, help="checkpoint 的 pretrained_model 目录")
    ap.add_argument("--name", type=str, help="这一轮评测的名字，用于数据集和 CSV，例如 act45_100k")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--start", type=int, default=1, help="从第几次开始（中途退出后接着测）")
    ap.add_argument("--quiet", action="store_true", help="不出声，只打印")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        args.name, args.policy, args.duration, args.quiet = "selftest", "none", 0.0, True
        # 第 1 次成功；第 2 次先重做再失败；第 3 次中途急停后开遥操作复位再重做、成功；然后退出
        script = ["start", "success", "start", "redo", "start", "fail", "start", "esc", "teleop", "teleop",
                  "redo", "start", "success", "esc"]
        keys = Keys(scripted=script)
    else:
        if not (args.policy and args.name):
            ap.error("需要 --policy 和 --name")
        keys = Keys()

    out_dir = ROOT / "outputs" / "eval" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = ROOT / "data" / "rollouts" / args.name
    csv_path = out_dir / "results.csv"
    new_csv = not csv_path.exists()
    fields = ["trial", "position", "orientation", "result", "started", "seconds", "policy", "note"]

    i = args.start
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_csv:
            w.writeheader()
        while i <= len(TRIALS):
            pos, ori = TRIALS[i - 1]
            print(f"\n== 第 {i}/{len(TRIALS)} 次：{pos}，{ori}", flush=True)
            say(f"Trial {i}. Press right arrow to start", args.quiet)
            keys.drain() if not args.selftest else None
            e = keys.get({"start", "teleop", "esc"})
            if e == "esc":
                say("Evaluation stopped", args.quiet)
                break
            if e == "teleop":
                teleop(keys, out_dir, args.quiet, args.selftest)
                continue
            started = datetime.now()
            cmd = [PY, "-c", "import time; time.sleep(0.2)"] if args.selftest else rollout_cmd(args, dataset_root)
            status = run_child(cmd, out_dir / f"trial_{i:02d}_{started:%H%M%S}.log", keys, args.quiet)
            seconds = round((datetime.now() - started).total_seconds(), 1)
            note = "急停" if status == "estop" else ""
            if status == "estop":
                say("Align the leader with the arm, press T to bring it back, then R to redo, or S or F to score", args.quiet)
            else:
                say("Success or fail? Press S or F. R to redo", args.quiet)
            while True:
                r = keys.get({"success", "fail", "redo", "teleop"})
                if r == "teleop":
                    teleop(keys, out_dir, args.quiet, args.selftest)
                    say("Press S, F, or R", args.quiet)
                    continue
                break
            if r == "redo":
                say("Redo", args.quiet)
                w.writerow(dict(trial=i, position=pos, orientation=ori, result="redo", started=f"{started:%H:%M:%S}",
                                seconds=seconds, policy=args.policy, note=note))
                f.flush()
                continue
            w.writerow(dict(trial=i, position=pos, orientation=ori, result=r, started=f"{started:%H:%M:%S}",
                            seconds=seconds, policy=args.policy, note=note))
            f.flush()
            say("Success" if r == "success" else "Fail", args.quiet)
            i += 1

    rows = [x for x in csv.DictReader(csv_path.open()) if x["result"] in ("success", "fail")]
    ok = sum(x["result"] == "success" for x in rows)
    print(f"\n已记录 {len(rows)} 次，成功 {ok} 次。结果：{csv_path}", flush=True)
    if len(rows) == len(TRIALS) and not args.selftest:
        say(f"Done. {ok} out of {len(rows)} succeeded", args.quiet)


if __name__ == "__main__":
    main()
