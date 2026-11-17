#!/usr/bin/env python3
"""真机评测：按固定的 20 个摆放位置逐次试抓，全程用键盘操作，结果写进 CSV。

每次试抓调用一次 LeRobot 的 lerobot-rollout（sentry 模式，把画面录下来，方便回放），
跑 --duration 秒后从臂自动回到起始姿势。每次试抓单独存一个录像目录，互不影响：
某一次急停或出错，只丢那一次的录像。ACT 和 π0.5 用同一份位置表，保证对比公平。

按键（全局监听，先点一下桌面空白处，免得按键落到别的程序里）：
    →    开始这一次试抓（药盒按语音报的位置摆好、从臂在起始姿势时再按）
    s    这次成功（药盒进了收纳盒）
    f    这次失败
    r    这次不算，重做（比如药盒没摆好）
    t    开 / 关遥操作（用主臂把从臂带回起始姿势；开之前先把主臂摆成和从臂差不多的姿势）
    Esc  试抓进行中：急停（立即结束程序，从臂停在原地保持力矩）；遥操作中：关遥操作；空闲时：退出

只有语音提示之后按的键才算：试抓过程中按的 s / f 会被丢掉，免得提前误判。

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/eval_trials.py --name act45_100k \\
        --policy outputs/train/act_medbox_45ep/checkpoints/100000/pretrained_model
    $V scripts/eval_trials.py --selftest    # 不连硬件、不监听键盘，用脚本化的按键检查流程

开始前：跑过 scripts/check_setup.py；从臂停在起始姿势。位置表见 TRIALS 和 docs/recording_guide.md。
结果：outputs/eval/<name>/results.csv；每次的日志在同一目录；录像在 data/rollouts/<name>/<trialNN_时间>/。
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
SLOW_MARK = "Record loop is running slower"  # LeRobot 每次控制循环超时都会打这一行
SLOW_WARN = 10  # 超时次数（不算第一步的预热）达到这个数就语音提醒：这一次的结果可能不可信
# 这个数是按每次 25 秒、每 30 步推理一次定的（第一轮每次 2–7 次）；每次更长时按时长等比放大
GUARD_MARK = "从臂读数保护已开启"  # scripts/rollout_safe.py 连上从臂后打的日志

# 位置名和录数据时的表一致（以操作者坐的位置为准）。方向：横放 = 长边和面前的桌沿平行，竖放 = 长边指向对面
TRIALS = [
    ("左上", "横放"), ("右上", "横放"), ("右下", "横放"), ("左下", "横放"), ("正中", "横放"),
    ("左上", "竖放"), ("右上", "竖放"), ("右下", "竖放"), ("左下", "竖放"), ("正中", "竖放"),
    ("左上", "斜放"), ("右上", "斜放"), ("右下", "斜放"), ("左下", "斜放"), ("正中", "斜放"),
    ("中偏左", "横放"), ("中偏右", "横放"), ("中偏左", "竖放"), ("中偏右", "竖放"), ("正中", "往另一边斜"),
]
# 语音只能说英文
SPOKEN = {"左上": "top left", "右上": "top right", "右下": "bottom right", "左下": "bottom left",
          "正中": "center", "中偏左": "center left", "中偏右": "center right",
          "横放": "horizontal", "竖放": "vertical", "斜放": "diagonal", "往另一边斜": "other diagonal"}


CHILDREN: list[subprocess.Popen] = []  # 启动过的试抓 / 遥操作子进程，本程序被中断时要一起正常结束


def stop_children() -> None:
    """本程序被 Ctrl+C 或 kill 中断时调用：还在跑的子进程发 SIGINT 正常结束（试抓会回到起始姿势，
    遥操作会停在原地），不留下没人管的遥操作。2026-11-16 停掉评测时遥操作子进程曾被留下继续运行。"""
    for p in CHILDREN:
        if p.poll() is None:
            p.send_signal(signal.SIGINT)
    for p in CHILDREN:
        try:
            p.wait(timeout=20)
        except subprocess.TimeoutExpired:
            print(f"子进程 {p.pid} 20 秒内没有退出", flush=True)


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
        """丢掉之前按下、还没处理的键（selftest 的脚本按键不能丢）。"""
        if self.scripted:
            return
        while not self.q.empty():
            self.q.get_nowait()


def rollout_cmd(args, rec_dir: Path, tag: str) -> list[str]:
    # rollout_safe.py = lerobot-rollout + 从臂读数保护（连上后的第一次读数可能是坏的，见该文件说明）
    return [PY, "scripts/rollout_safe.py", "--config_path=configs/rollout_base.yaml",
            f"--policy.path={args.policy}", f"--duration={args.duration}", "--play_sounds=false",
            "--strategy.type=sentry", f"--dataset.repo_id=local/rollout_{args.name}_{tag}",
            f"--dataset.root={rec_dir}", f"--dataset.single_task={TASK}", "--dataset.fps=15",
            "--dataset.push_to_hub=false", "--dataset.video=true"] + policy_overrides(args)


def policy_overrides(args) -> list[str]:
    """推理时可以改、不用重新训练的 ACT 设置。"""
    if args.ensemble is not None:  # 时间集成：每一步都重新推理，把几个动作块对同一时刻的预测加权平均
        return [f"--policy.temporal_ensemble_coeff={args.ensemble}", "--policy.n_action_steps=1"]
    if args.n_action_steps is not None:  # 每个动作块只执行前 N 步就重新推理
        return [f"--policy.n_action_steps={args.n_action_steps}"]
    return []


def fake_rollout(n: int) -> list[str]:
    """selftest 用的假试抓：第 1 次打 12 行超时警告，第 6 次以错误码退出，其余正常结束。"""
    warn = f"print('WARNING {SLOW_MARK} (5.0 Hz)');" * (12 if n == 1 else 1)
    code = 1 if n == 6 else 0
    end = "print('Rollout finished');" if code == 0 else "print('Traceback: fake crash');"
    return [PY, "-c", f"import sys, time; {warn} time.sleep(0.2); {end} sys.exit({code})"]


def run_child(cmd: list[str], log_path: Path, keys: Keys, quiet: bool) -> tuple[str, int]:
    """运行一次试抓，返回 (状态, 超时次数)。状态：'done' 正常结束，'estop' 急停，'error' 程序出错退出。"""
    with log_path.open("w") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                             env={**os.environ, "PYTHONUNBUFFERED": "1"})
        CHILDREN.append(p)
        keys.estop.clear()
        while p.poll() is None:
            if keys.scripted and keys.peek() == "estop":  # selftest：脚本里用 "estop" 表示试抓中按 Esc
                keys.get({"estop"})
                keys.estop.set()
            if keys.estop.is_set():
                p.send_signal(signal.SIGKILL)  # 不走收尾流程：不回位，从臂停在原地保持力矩
                p.wait()
                say("Emergency stop", quiet)
                return "estop", 0
            time.sleep(0.05)
    text = log_path.read_text(errors="replace")
    slow = max(0, text.count(SLOW_MARK) - 1)  # 第一步要预热 GPU，必然超时一次，不算
    if p.returncode != 0 or "Rollout finished" not in text:
        return "error", slow
    if not keys.scripted and GUARD_MARK not in text:  # 读数保护没生效：结果不可信，也说明命令被改坏了
        return "error", slow
    return "done", slow


def teleop(keys: Keys, log_dir: Path, quiet: bool, selftest: bool) -> None:
    # 遥操作一启动，从臂会快速转到主臂的姿势，所以按 T 之前主臂要先摆成和从臂差不多
    say("Teleop on. Press T again when the arm is at the start pose", quiet)
    cmd = [PY, "-c", "import time; time.sleep(0.2)"] if selftest else \
        [PY, "-m", "lerobot.scripts.lerobot_teleoperate", "--config_path=configs/teleop_practice.yaml"]
    with (log_dir / f"teleop_{datetime.now():%H%M%S}.log").open("w") as log:
        p = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        CHILDREN.append(p)
        keys.get({"teleop", "esc"})
        if p.poll() is None:
            p.send_signal(signal.SIGINT)  # 正常退出：从臂停在当前姿势、保持力矩
            p.wait(timeout=15)
    keys.estop.clear()
    say("Teleop off", quiet)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=str, help="checkpoint 的 pretrained_model 目录")
    ap.add_argument("--name", type=str, help="这一轮评测的名字，用于录像目录和 CSV，例如 act45_100k")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--ensemble", type=float, default=None,
                    help="ACT 时间集成系数（如 0.01）：每一步都重新推理并加权平均；不用重新训练")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="每个动作块执行几步就重新推理（训练时是 30，即 2 秒）；不用重新训练")
    ap.add_argument("--start", type=int, default=1, help="从第几次开始（中途退出后接着测）")
    ap.add_argument("--quiet", action="store_true", help="不出声，只打印")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        args.name, args.policy, args.duration, args.quiet = "selftest", "none", 0.0, True
        # 第 1 次成功（但控制循环慢，要记下来）；第 2 次先重做再失败；第 3 次急停 → 遥操作复位 → 重做 → 成功；
        # 第 4 次程序出错（不计分、停在第 4 次）；然后退出
        script = ["start", "success", "start", "redo", "start", "fail", "start", "estop", "teleop", "teleop",
                  "redo", "start", "success", "start", "esc"]
        keys = Keys(scripted=script)
    else:
        if not (args.policy and args.name):
            ap.error("需要 --policy 和 --name")
        keys = Keys()

    if args.ensemble is not None and args.n_action_steps is not None:
        ap.error("--ensemble 和 --n-action-steps 只能选一个")
    policy_label = " ".join([args.policy] + policy_overrides(args))  # 写进 CSV 的 policy 列

    out_dir = ROOT / "outputs" / "eval" / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_root = ROOT / "data" / "rollouts" / args.name
    rec_root.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    new_csv = not csv_path.exists()
    fields = ["trial", "position", "orientation", "result", "started", "seconds", "slow_steps", "note",
              "recording", "policy"]

    i, n_rollouts = args.start, 0
    with csv_path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new_csv:
            w.writeheader()

        def record(result: str, note: str = "") -> None:
            w.writerow(dict(trial=i, position=pos, orientation=ori, result=result, started=f"{started:%H:%M:%S}",
                            seconds=seconds, slow_steps=slow, note=note,
                            recording=str(rec_dir.relative_to(ROOT)), policy=policy_label))
            f.flush()

        while i <= len(TRIALS):
            pos, ori = TRIALS[i - 1]
            print(f"\n== 第 {i}/{len(TRIALS)} 次：{pos}，{ori}", flush=True)
            keys.drain()
            say(f"Trial {i}. {SPOKEN[pos]}, {SPOKEN[ori]}. Press right arrow to start", args.quiet)
            e = keys.get({"start", "teleop", "esc"})
            if e == "esc":
                say("Evaluation stopped", args.quiet)
                break
            if e == "teleop":
                teleop(keys, out_dir, args.quiet, args.selftest)
                continue

            started = datetime.now()
            tag = f"trial{i:02d}_{started:%H%M%S}"
            while (rec_root / tag).exists() or (out_dir / f"{tag}.log").exists():  # 同一秒内重开时别撞名
                tag += "b"
            rec_dir = rec_root / tag
            n_rollouts += 1
            cmd = fake_rollout(n_rollouts) if args.selftest else rollout_cmd(args, rec_dir, tag)
            status, slow = run_child(cmd, out_dir / f"{tag}.log", keys, args.quiet)
            seconds = round((datetime.now() - started).total_seconds(), 1)
            print(f"   结束：{status}，用时 {seconds}s，控制循环超时 {slow} 次", flush=True)
            keys.drain()  # 试抓过程中按的键不算

            if status == "error":
                record("error", "程序出错退出，见日志")
                say("The rollout program failed. Tell Claude", args.quiet)
                continue

            notes = []
            if status == "estop":
                notes.append("急停")
                say("Align the leader with the arm, press T to bring it back, then R to redo, or S or F to score",
                    args.quiet)
            else:
                if slow >= SLOW_WARN * max(args.duration, 25.0) / 25.0:
                    notes.append(f"控制循环超时 {slow} 次")
                    say(f"Warning: the control loop was slow {slow} times. Consider redo", args.quiet)
                say("Success or fail? Press S or F. R to redo", args.quiet)
            while True:
                r = keys.get({"success", "fail", "redo", "teleop"})
                if r == "teleop":
                    teleop(keys, out_dir, args.quiet, args.selftest)
                    keys.drain()
                    say("Press S, F, or R", args.quiet)
                    continue
                break
            record(r, "；".join(notes))
            if r == "redo":
                say("Redo", args.quiet)
                continue
            say("Success" if r == "success" else "Fail", args.quiet)
            i += 1

    rows = [x for x in csv.DictReader(csv_path.open()) if x["result"] in ("success", "fail")]
    ok = sum(x["result"] == "success" for x in rows)
    print(f"\n已记录 {len(rows)} 次，成功 {ok} 次。结果：{csv_path}", flush=True)
    if len(rows) == len(TRIALS) and not args.selftest:
        say(f"Done. {ok} out of {len(rows)} succeeded", args.quiet)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))  # kill 也走下面的收尾
    try:
        main()
    finally:
        stop_children()
