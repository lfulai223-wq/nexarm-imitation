#!/usr/bin/env python3
"""真机评测：按固定的 20 个摆放位置逐次试抓，全程用键盘操作，结果写进 CSV。

每次试抓调用一次 LeRobot 的 lerobot-rollout（sentry 模式，把画面录下来，方便回放），
跑 --duration 秒后从臂自动回到起始姿势。每次试抓单独存一个录像目录，互不影响：
某一次急停或出错，只丢那一次的录像。ACT 和 π0.5 用同一份位置表，保证对比公平。

两个模型对比（加 --policy-b 和 --name-b）：每个摆放位置连着测两次，两个模型各一次。
单数次先测 A（--policy）再测 B，双数次反过来。这样一个小时里光线、相机这些随时间的变化对两个模型影响一样，
操作者也是同一个位置连摆两次。语音只说第一轮、第二轮，不说是哪个模型。
两个模型的结果分别写进各自的 CSV，格式和只测一个模型时一样。中途退出后用同样的命令重新运行，
已经判过成功或失败的会自动跳过，从断开的地方接着测；同一个名字下已有别的模型或别的时长的结果时拒绝运行。

按键（全局监听，先点一下桌面空白处，免得按键落到别的程序里）：
    →    开始这一次试抓（药盒按语音报的位置摆好、从臂在起始姿势时再按）
    s    这次成功（药盒进了收纳盒）
    f    这次失败
    r    这次不算，重做（比如药盒没摆好）
    t    开 / 关遥操作（用主臂把从臂带回起始姿势；开之前先把主臂摆成和从臂差不多的姿势）
    Esc  试抓进行中：急停（立即结束程序，从臂停在原地保持力矩）；遥操作中：关遥操作；空闲时：退出

只有语音提示之后按的键才算：试抓过程中按的 s / f 会被丢掉，免得提前误判。
试抓过程中手不要伸进去：药盒掉了、歪了也不摆回去，等它跑完按 f。有人手帮过的按失败算（2026-11-22 定）。

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/eval_trials.py --name act45_100k \\
        --policy outputs/train/act_medbox_45ep/checkpoints/100000/pretrained_model
    $V scripts/eval_trials.py --duration 40 \\
        --name act45_100k_ab --policy outputs/train/act_medbox_45ep/checkpoints/100000/pretrained_model \\
        --name-b act102_100k_ab --policy-b outputs/train/act_medbox_102ep/checkpoints/100000/pretrained_model
    $V scripts/eval_trials.py --duration 40 --n-action-steps 30 \\
        --name act102_100k_vs_pi05 --policy outputs/train/act_medbox_102ep/checkpoints/100000/pretrained_model \\
        --name-b pi05_12k_vs_act102 --policy-b outputs/train/pi05_lora_102ep/checkpoints/012000/pretrained_model
    $V scripts/eval_trials.py --selftest    # 不连硬件、不监听键盘，用脚本化的按键检查两种流程，核对 CSV

π0.5（LoRA 微调，2026-11-18 起）：rollout_safe.py 加载时跳过随机初始化、把 LoRA 合并进权重，
日志里没有合并的记录就记 error。π0.5 每次重新预测约 0.25 秒，超过一个控制周期，必然超时一次，
提醒门槛会自动加上这些次数（按 时长 × 15 / 每块执行步数 算）。--n-action-steps 对同步推理的模型生效，ACT 训练时就是 30。
加 --rtc 用 RTC 异步推理：后台预测、不停顿，参数见 inference_overrides。RTC 只加在流匹配的模型（π0.5）上；
两个模型交替测时，另一个不是流匹配的模型（ACT）照常同步推理（2026-11-20 起）。

    $V scripts/eval_trials.py --duration 40 --rtc \\
        --name pi05_12k_rtc --policy outputs/train/pi05_lora_102ep/checkpoints/012000/pretrained_model

第二份位置表（--placements fixed2，2026-11-20 起）：20 个新的固定摆放，横、竖、斜、往另一边斜的个数和第一份一样（7 / 7 / 5 / 1），
位置换成起点区四条边的中点和"正中与四个角之间"，见 TRIALS2。第二轮 ACT-102 对 π0.5 RTC 交替测用它：

    $V scripts/eval_trials.py --duration 40 --rtc --placements fixed2 \\
        --name act102_100k_r2 --policy outputs/train/act_medbox_102ep/checkpoints/100000/pretrained_model \\
        --name-b pi05_12k_rtc_r2 --policy-b outputs/train/pi05_lora_102ep/checkpoints/012000/pretrained_model

随机摆放（--free N，2026-11-18 起）：不用位置表，每次由操作者换一个新位置、新角度随便摆，共 N 次，只测一个模型。
CSV 的位置、方向两列记"随机"，实际摆在哪、转了多少度，事后从录像第一帧量。

    $V scripts/eval_trials.py --duration 40 --rtc --free 20 \\
        --name pi05_12k_rtc_free --policy outputs/train/pi05_lora_102ep/checkpoints/012000/pretrained_model

开始前：跑过 scripts/check_setup.py；从臂停在起始姿势。位置表见 TRIALS 和 docs/recording_guide.md。
结果：outputs/eval/<name>/results.csv；每次的日志在同一目录；录像在 data/rollouts/<name>/<trialNN_时间>/。
逐次的阶段分析和两个模型的配对比较：scripts/eval_stages.py。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import IO

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
TASK = "Pick up the medicine box and put it in the bin."
SLOW_MARK = "Record loop is running slower"  # LeRobot 每次控制循环超时都会打这一行
SLOW_WARN = 10  # 超时次数（不算第一步的预热）达到这个数就语音提醒：这一次的结果可能不可信
# 这个数是按每次 25 秒、每 30 步推理一次定的（第一轮每次 2–7 次）；每次更长时按时长等比放大
GUARD_MARK = "从臂读数保护已开启"  # scripts/rollout_safe.py 连上从臂后打的日志
MERGE_MARK = "LoRA 已合并进权重"  # rollout_safe.py 合并 LoRA 后打的日志（π0.5 这类 LoRA 微调的模型才有）

# 位置名和录数据时的表一致（以操作者坐的位置为准）。方向：横放 = 长边和面前的桌沿平行，竖放 = 长边指向对面
TRIALS = [
    ("左上", "横放"), ("右上", "横放"), ("右下", "横放"), ("左下", "横放"), ("正中", "横放"),
    ("左上", "竖放"), ("右上", "竖放"), ("右下", "竖放"), ("左下", "竖放"), ("正中", "竖放"),
    ("左上", "斜放"), ("右上", "斜放"), ("右下", "斜放"), ("左下", "斜放"), ("正中", "斜放"),
    ("中偏左", "横放"), ("中偏右", "横放"), ("中偏左", "竖放"), ("中偏右", "竖放"), ("正中", "往另一边斜"),
]
# 第二份位置表（2026-11-20）：和第一份不重复的位置。以第一份的四个角（左上、右上、右下、左下附近）为准：
# "X边中间" 是相邻两个角的正中间，"中偏X" 是正中和角 X 的正中间。方向个数和第一份一样（横 7、竖 7、斜 5、往另一边斜 1），
# 右侧（右边中间、中偏右上、中偏右下）7 个，和第一份右侧的 8 个差不多：这一片在支架相机画面下沿，第一轮里最难。
# 顺序事先打乱、写死：相邻两次的位置和方向都不一样，免得某个方向全挤在评测的开头或结尾。
TRIALS2 = [
    ("上边中间", "横放"), ("右边中间", "竖放"), ("下边中间", "斜放"), ("左边中间", "横放"), ("中偏右上", "往另一边斜"),
    ("右边中间", "横放"), ("中偏左下", "竖放"), ("上边中间", "斜放"), ("中偏右下", "横放"), ("左边中间", "竖放"),
    ("下边中间", "横放"), ("中偏左上", "竖放"), ("右边中间", "斜放"), ("中偏右上", "横放"), ("上边中间", "竖放"),
    ("左边中间", "斜放"), ("中偏右下", "竖放"), ("中偏左下", "斜放"), ("下边中间", "竖放"), ("中偏左上", "横放"),
]
# 鲁棒性测试的 8 个位置（2026-11-22）：第二份位置表里 11-22 两个模型都成功的第 2、4、7、9、12、13、15、16 个
# （不用第 19 个：药盒就在起始姿势的夹爪正下方，太容易）。竖 4、横 2、斜 2，顺序照第二份表，相邻两次位置和方向都不一样。
ROBUST8 = [TRIALS2[i - 1] for i in (2, 4, 7, 9, 12, 13, 15, 16)]
PLACEMENT_SETS = {"fixed": TRIALS, "fixed2": TRIALS2, "robust8": ROBUST8}
# 鲁棒性测试的条件：每一次报完位置后语音提醒怎么布置（clean 是原条件，不提醒）
CONDITIONS = {"clean": "",
              "card": "Card flat beside the box, two fingers away, on your side. ",
              "lamp": "Side lamp on. "}
# 语音只能说英文
SPOKEN = {"左上": "top left", "右上": "top right", "右下": "bottom right", "左下": "bottom left",
          "正中": "center", "中偏左": "center left", "中偏右": "center right",
          "上边中间": "top edge, middle", "右边中间": "right edge, middle", "下边中间": "bottom edge, middle",
          "左边中间": "left edge, middle", "中偏左上": "center, toward top left", "中偏右上": "center, toward top right",
          "中偏右下": "center, toward bottom right", "中偏左下": "center, toward bottom left",
          "横放": "horizontal", "竖放": "vertical", "斜放": "diagonal", "往另一边斜": "other diagonal"}
FREE = ("随机", "随机")  # --free 时每一次的位置、方向


def trial_list(args) -> list[tuple[str, str]]:
    return [FREE] * args.free if args.free else PLACEMENT_SETS[args.placements]

FIELDS = ["trial", "position", "orientation", "result", "started", "seconds", "slow_steps", "note",
          "recording", "policy"]

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
            if self.scripted and self.q.empty():  # 自测：脚本写错时报错，不要一直等下去
                raise RuntimeError(f"自测脚本的按键用完了，还在等 {sorted(allowed)}")
            left = None if end is None else max(0.0, end - time.monotonic())
            try:
                e = self.q.get(timeout=left)
            except queue.Empty:
                return None
            if e in allowed:
                return e
            if self.scripted:
                raise RuntimeError(f"自测脚本的按键对不上：在等 {sorted(allowed)}，脚本给的是 {e}")

    def peek(self) -> str | None:
        with self.q.mutex:
            return self.q.queue[0] if self.q.queue else None

    def drain(self) -> None:
        """丢掉之前按下、还没处理的键（selftest 的脚本按键不能丢）。"""
        if self.scripted:
            return
        while not self.q.empty():
            self.q.get_nowait()


@dataclass
class Model:
    """一个被测模型：自己的名字、录像目录和结果 CSV。"""

    name: str
    policy: str
    label: str  # 写进 CSV 的 policy 列：模型目录、每次时长和推理时改过的设置
    out_dir: Path
    rec_root: Path
    use_peft: bool = False  # LoRA 微调的模型：每次都要确认 rollout_safe.py 把 LoRA 合并进了权重
    expected_slow: int = 0  # 推理本身就超过一个控制周期的模型（π0.5），每次重新预测必然超时一次，提醒门槛要加上这些
    rtc: bool = False  # 这个模型用 RTC 异步推理（给了 --rtc 而且是流匹配的模型）
    kind: str = ""  # config.json 里的模型类型（act、pi05），读不到为空
    file: IO[str] | None = None
    writer: csv.DictWriter | None = None

    @property
    def csv_path(self) -> Path:
        return self.out_dir / "results.csv"

    def scored(self) -> list[dict[str, str]]:
        """CSV 里已经判了成功或失败的行（重做、出错的不算）。"""
        if not self.csv_path.exists():
            return []
        with self.csv_path.open(newline="") as f:
            return [r for r in csv.DictReader(f) if r["result"] in ("success", "fail")]


def policy_info(policy: str, base: Path) -> dict:
    """读检查点的 config.json：模型类型、是不是 LoRA 微调、每个动作块执行几步。读不到（自测的假模型）就返回空。"""
    p = Path(policy) if Path(policy).is_absolute() else base / policy
    try:
        return json.loads((p / "config.json").read_text())
    except (OSError, ValueError):
        return {}


def make_model(name: str, policy: str, args, base: Path) -> Model:
    """只建对象，不建目录（run_eval 检查完参数再建）。"""
    info = policy_info(policy, base)
    kind = info.get("type", "")
    rtc = bool(args.rtc) and kind in FLOW_POLICIES  # RTC 只加在流匹配的模型上，ACT 照常同步推理
    label = " ".join([policy, f"--duration={args.duration:g}"] + policy_overrides(args, rtc) + inference_overrides(rtc))
    expected_slow = 0
    if kind == "pi05" and not rtc:  # 同步推理：重新预测一个动作块约 0.25 秒，每次重新预测超时一次
        n_steps = args.n_action_steps or info.get("n_action_steps") or 50
        expected_slow = math.ceil(args.duration * 15 / n_steps)
    return Model(name, policy, label, base / "outputs" / "eval" / name, base / "data" / "rollouts" / name,
                 use_peft=bool(info.get("use_peft")), expected_slow=expected_slow, rtc=rtc, kind=kind)


def rollout_cmd(args, m: Model, rec_dir: Path, tag: str) -> list[str]:
    # rollout_safe.py = lerobot-rollout + 从臂读数保护（连上后的第一次读数可能是坏的，见该文件说明）
    return [PY, "scripts/rollout_safe.py", "--config_path=configs/rollout_base.yaml",
            f"--policy.path={m.policy}", f"--duration={args.duration}", "--play_sounds=false",
            "--strategy.type=sentry", f"--dataset.repo_id=local/rollout_{m.name}_{tag}",
            f"--dataset.root={rec_dir}", f"--dataset.single_task={TASK}", "--dataset.fps=15",
            "--dataset.push_to_hub=false", "--dataset.video=true"] + policy_overrides(args, m.rtc) + inference_overrides(m.rtc)


FLOW_POLICIES = ("pi05", "pi0", "smolvla")  # LeRobot 的 RTC 只支持这些流匹配模型


def inference_overrides(rtc: bool) -> list[str]:
    """RTC 异步推理（实时分块）：后台线程预测下一块，机械臂不停顿；新的一块开头按上一块还没执行完的动作引导，衔接平滑。
    队列剩 20 步时开始预测下一块：动作块 50 步，所以每执行约 30 步（2 秒）看一次新画面，和同步推理、ACT 的节奏一样。
    execution_horizon 10、max_guidance_weight 10、EXP 是 LeRobot RTC 文档的建议值。"""
    if not rtc:
        return []
    return ["--inference.type=rtc", "--inference.rtc.execution_horizon=10", "--inference.rtc.max_guidance_weight=10.0",
            "--inference.rtc.prefix_attention_schedule=EXP", "--inference.queue_threshold=20"]


def policy_overrides(args, rtc: bool = False) -> list[str]:
    """推理时可以改、不用重新训练的设置，只对同步推理的模型生效（RTC 时每块执行多少步由队列决定）。"""
    if rtc:
        return []
    if args.ensemble is not None:  # 时间集成：每一步都重新推理，把几个动作块对同一时刻的预测加权平均
        return [f"--policy.temporal_ensemble_coeff={args.ensemble}", "--policy.n_action_steps=1"]
    if args.n_action_steps is not None:  # 每个动作块只执行前 N 步就重新推理
        return [f"--policy.n_action_steps={args.n_action_steps}"]
    return []


def fake_rollout(n: int) -> list[str]:
    """selftest 用的假试抓：每一段自测里第 1 次打 12 行超时警告，第 6 次以错误码退出，其余正常结束。"""
    warn = f"print('WARNING {SLOW_MARK} (5.0 Hz)');" * (12 if n == 1 else 1)
    code = 1 if n == 6 else 0
    end = "print('Rollout finished');" if code == 0 else "print('Traceback: fake crash');"
    return [PY, "-c", f"import sys, time; {warn} time.sleep(0.2); {end} sys.exit({code})"]


def run_child(cmd: list[str], log_path: Path, keys: Keys, quiet: bool,
              marks: tuple[str, ...] = ()) -> tuple[str, int]:
    """运行一次试抓，返回 (状态, 超时次数)。状态：'done' 正常结束，'estop' 急停，'error' 程序出错退出。
    marks：日志里必须出现的行（除了读数保护），缺了也记成 error。"""
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
    if not keys.scripted and any(mk not in text for mk in marks):
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


def run_trial(args, keys: Keys, m: Model, i: int, place: tuple[str, str], header: str, prompt: str,
              base: Path, n_rollouts: list[int]) -> bool:
    """在第 i 个摆放位置 place（位置, 方向）用模型 m 试抓，判了成功或失败就返回 True；在开始提示时按 Esc 返回 False。
    重做、程序出错都在这里面重来，不换位置也不换模型。"""
    pos, ori = place
    while True:
        print(header, flush=True)
        keys.drain()
        say(prompt, args.quiet)
        e = keys.get({"start", "teleop", "esc"})
        if e == "esc":
            return False
        if e == "teleop":
            teleop(keys, m.out_dir, args.quiet, args.selftest)
            continue

        # 试抓过程中不碰药盒：掉了、歪了也不摆回去，有人帮过就按失败算（2026-11-22 复查录像后定）
        say("Running. Hands off until it stops", args.quiet)
        started = datetime.now()
        tag = f"trial{i:02d}_{started:%H%M%S}"
        while (m.rec_root / tag).exists() or (m.out_dir / f"{tag}.log").exists():  # 同一秒内重开时别撞名
            tag += "b"
        rec_dir = m.rec_root / tag
        n_rollouts[0] += 1
        cmd = fake_rollout(n_rollouts[0]) if args.selftest else rollout_cmd(args, m, rec_dir, tag)
        status, slow = run_child(cmd, m.out_dir / f"{tag}.log", keys, args.quiet,
                                 marks=(MERGE_MARK,) if m.use_peft else ())
        seconds = round((datetime.now() - started).total_seconds(), 1)
        print(f"   结束：{status}，用时 {seconds}s，控制循环超时 {slow} 次", flush=True)
        keys.drain()  # 试抓过程中按的键不算

        def record(result: str, note: str = "") -> None:
            m.writer.writerow(dict(trial=i, position=pos, orientation=ori, result=result,
                                   started=f"{started:%H:%M:%S}", seconds=seconds, slow_steps=slow, note=note,
                                   recording=str(rec_dir.relative_to(base)), policy=m.label))
            m.file.flush()

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
            if slow >= SLOW_WARN * max(args.duration, 25.0) / 25.0 + m.expected_slow:
                notes.append(f"控制循环超时 {slow} 次")
                say(f"Warning: the control loop was slow {slow} times. Consider redo", args.quiet)
            say("Success or fail? Press S or F. R to redo", args.quiet)
        while True:
            r = keys.get({"success", "fail", "redo", "teleop"})
            if r == "teleop":
                teleop(keys, m.out_dir, args.quiet, args.selftest)
                keys.drain()
                say("Press S, F, or R", args.quiet)
                continue
            break
        record(r, "；".join(notes))
        if r == "redo":
            say("Redo", args.quiet)
            continue
        say("Success" if r == "success" else "Fail", args.quiet)
        return True


def run_eval(args, keys: Keys, base: Path = ROOT) -> None:
    models = [make_model(args.name, args.policy, args, base)]
    if args.policy_b:
        models.append(make_model(args.name_b, args.policy_b, args, base))
    ab = len(models) == 2
    if args.rtc and not any(m.rtc for m in models):
        raise SystemExit(f"--rtc 只能用于流匹配的模型（{', '.join(FLOW_POLICIES)}），"
                         f"这次的模型是 {', '.join(m.kind or '未知' for m in models)}")
    if (args.ensemble is not None or args.n_action_steps is not None) and all(m.rtc for m in models):
        raise SystemExit("--rtc 时每块执行多少步由队列决定（剩 20 步就预测下一块），不能再给 --ensemble 或 --n-action-steps")
    for m in models:
        m.out_dir.mkdir(parents=True, exist_ok=True)
        m.rec_root.mkdir(parents=True, exist_ok=True)
        print(f"{m.name}：{m.kind or '未知类型'}，{'RTC 异步推理' if m.rtc else '同步推理'}；{m.label}", flush=True)
    trials = trial_list(args)

    # 交替测时，已经判过的自动跳过；所以同一个名字下不能有别的模型、别的时长或别的位置表的结果，否则会被当成这次的
    done: dict[str, set[int]] = {m.name: set() for m in models}
    if ab:
        for m in models:
            rows = m.scored()
            other = sorted({r["policy"] for r in rows} - {m.label})
            if other:
                raise SystemExit(f"{m.csv_path} 里已经有别的模型或别的时长的结果（{other[0]}），"
                                 f"这次是 {m.label}。换一个名字再测")
            for r in rows:
                t = int(r["trial"])
                if t > len(trials) or (r["position"], r["orientation"]) != trials[t - 1]:
                    want = "（这份表没有这一次）" if t > len(trials) else "".join(trials[t - 1])
                    raise SystemExit(f"{m.csv_path} 第 {t} 次记的是 {r['position']}{r['orientation']}，"
                                     f"这次用的位置表第 {t} 个是 {want}。换一个名字再测")
            done[m.name] = {int(r["trial"]) for r in rows}
            if done[m.name]:
                print(f"{m.name} 已经测过 {len(done[m.name])} 次，这些会跳过", flush=True)

    n_rollouts = [0]
    with contextlib.ExitStack() as stack:
        for m in models:
            new_csv = not m.csv_path.exists()
            m.file = stack.enter_context(m.csv_path.open("a", newline=""))
            m.writer = csv.DictWriter(m.file, fieldnames=FIELDS)
            if new_csv:
                m.writer.writeheader()

        stopped = False
        setup = CONDITIONS[args.condition]
        for i in range(args.start, len(trials) + 1):
            pos, ori = trials[i - 1]
            order = models if i % 2 == 1 else models[::-1]  # 交替测：单数次先 A，双数次先 B
            for k, m in enumerate(order):
                if i in done[m.name]:
                    continue
                if ab:
                    header = f"\n== 第 {i}/{len(trials)} 次，第 {k + 1} 轮（{m.name}）：{pos}，{ori}"
                    prompt = (f"Trial {i}, first run. {SPOKEN[pos]}, {SPOKEN[ori]}. {setup}Press right arrow to start"
                              if k == 0 else
                              f"Trial {i}, second run. Same place: {SPOKEN[pos]}, {SPOKEN[ori]}. "
                              f"{setup}Press right arrow to start")
                elif args.free:
                    header = f"\n== 第 {i}/{len(trials)} 次：随机摆放"
                    prompt = f"Trial {i}. Put the box somewhere new, at any angle. Press right arrow to start"
                else:
                    header = f"\n== 第 {i}/{len(trials)} 次：{pos}，{ori}"
                    prompt = f"Trial {i}. {SPOKEN[pos]}, {SPOKEN[ori]}. {setup}Press right arrow to start"
                if not run_trial(args, keys, m, i, (pos, ori), header, prompt, base, n_rollouts):
                    say("Evaluation stopped", args.quiet)
                    stopped = True
                    break
            if stopped:
                break

    counts = []
    for m in models:
        rows = m.scored()
        ok = sum(r["result"] == "success" for r in rows)
        counts.append((len(rows), ok))
        print(f"\n{m.name}：已记录 {len(rows)} 次，成功 {ok} 次。结果：{m.csv_path}", flush=True)
    if not args.selftest and all(n == len(trials) for n, _ in counts):
        if ab:
            say(f"Done. All {2 * len(trials)} runs are scored", args.quiet)
        else:
            say(f"Done. {counts[0][1]} out of {counts[0][0]} succeeded", args.quiet)


def selftest() -> None:
    """不连硬件、不监听键盘：用脚本化的按键跑一遍只测一个模型和两个模型交替，核对写进 CSV 的结果。
    假试抓见 fake_rollout：每一段里第 1 次控制循环慢（要提醒），第 6 次程序出错（记 error，同一次重来）。"""
    with tempfile.TemporaryDirectory(prefix="eval_selftest_") as tmp:
        base = Path(tmp)

        def ns(**kw):
            d = dict(policy=None, name=None, policy_b=None, name_b=None, duration=0.0, ensemble=None,
                     n_action_steps=None, rtc=False, free=None, placements="fixed", condition="clean", start=1, quiet=True,
                     selftest=True)
            return argparse.Namespace(**{**d, **kw})

        def rows(name: str) -> list[dict[str, str]]:
            with (base / "outputs" / "eval" / name / "results.csv").open(newline="") as f:
                return list(csv.DictReader(f))

        def expect(name: str, want: list[tuple[int, str]]) -> None:
            got = [(int(r["trial"]), r["result"]) for r in rows(name)]
            assert got == want, f"{name}：应该是 {want}，实际是 {got}"

        def note(name: str, trial: int, result: str) -> str:
            return next(r["note"] for r in rows(name) if int(r["trial"]) == trial and r["result"] == result)

        print("== 自测 1：只测一个模型", flush=True)
        # 第 1 次成功（但控制循环慢，要记下来）；第 2 次先重做再失败；第 3 次急停 → 遥操作复位 → 重做 → 成功；
        # 第 4 次程序出错（不计分、停在第 4 次）；然后退出
        run_eval(ns(name="single", policy="p1"), Keys(scripted=[
            "start", "success", "start", "redo", "start", "fail", "start", "estop", "teleop", "teleop",
            "redo", "start", "success", "start", "esc"]), base)
        expect("single", [(1, "success"), (2, "redo"), (2, "fail"), (3, "redo"), (3, "success"), (4, "error")])
        assert "控制循环超时" in note("single", 1, "success") and note("single", 3, "redo") == "急停"

        ab = dict(name="a", policy="pa", name_b="b", policy_b="pb")
        print("\n== 自测 2：两个模型交替，第一段（单数次先 A 后 B，双数次先 B 后 A）", flush=True)
        run_eval(ns(**ab), Keys(scripted=[
            "start", "success",                             # 1A（控制循环慢）
            "start", "redo", "start", "fail",               # 1B 先重做再失败
            "start", "estop", "teleop", "teleop", "fail",   # 2B 急停，遥操作复位后判失败
            "start", "success",                             # 2A
            "start", "start", "success",                    # 3A 第一次程序出错，重来后成功
            "esc"]), base)                                  # 3B 还没开始就退出
        print("\n== 自测 3：同样的命令再运行一次，判过的跳过，从 3B 接着测", flush=True)
        run_eval(ns(**ab), Keys(scripted=[
            "start", "success",                             # 3B（这一段的第 1 次，控制循环慢）
            "start", "fail",                                # 4B（双数次先 B）
            "esc"]), base)                                  # 4A 还没开始就退出
        expect("a", [(1, "success"), (2, "success"), (3, "error"), (3, "success")])
        expect("b", [(1, "redo"), (1, "fail"), (2, "fail"), (3, "success"), (4, "fail")])
        assert "控制循环超时" in note("a", 1, "success") and note("b", 2, "fail") == "急停"
        assert all(r["policy"] == "pa --duration=0" for r in rows("a"))

        print("\n== 自测 4：同一个名字换了模型，应该拒绝", flush=True)
        try:
            run_eval(ns(name="a", policy="pother", name_b="b", policy_b="pb"), Keys(scripted=[]), base)
        except SystemExit as e:
            print(f"拒绝了：{e}", flush=True)
        else:
            raise AssertionError("换了模型却没有拒绝")

        print("\n== 自测 5：随机摆放 3 次（不用位置表，测完 3 次自己结束）", flush=True)
        run_eval(ns(name="free", policy="pf", free=3), Keys(scripted=[
            "start", "success", "start", "redo", "start", "fail", "start", "success"]), base)
        expect("free", [(1, "success"), (2, "redo"), (2, "fail"), (3, "success")])
        assert all((r["position"], r["orientation"]) == FREE for r in rows("free"))

        # 假模型目录：只有 config.json，用来认模型类型
        for p, cfg in (("pact", {"type": "act", "n_action_steps": 30}), ("pact2", {"type": "act", "n_action_steps": 30}),
                       ("ppi", {"type": "pi05", "use_peft": True, "n_action_steps": 50})):
            (base / p).mkdir()
            (base / p / "config.json").write_text(json.dumps(cfg))
        rtc_flag = "--inference.type=rtc"

        print("\n== 自测 6：ACT 和 π0.5 交替，--rtc 只加在 π0.5 上，用第二份位置表", flush=True)
        mix = dict(name="act_r2", policy="pact", name_b="pi_r2", policy_b="ppi", rtc=True, placements="fixed2")
        run_eval(ns(**mix), Keys(scripted=["start", "success", "start", "fail", "esc"]), base)  # 第 1 次 A、B；第 2 次没开始就退出
        expect("act_r2", [(1, "success")])
        expect("pi_r2", [(1, "fail")])
        ra, rb = rows("act_r2")[0], rows("pi_r2")[0]
        assert (ra["position"], ra["orientation"]) == TRIALS2[0] == (rb["position"], rb["orientation"])
        assert rtc_flag not in ra["policy"] and rtc_flag in rb["policy"], (ra["policy"], rb["policy"])
        ma, mb = (make_model("x", p, ns(**mix), base) for p in ("pact", "ppi"))
        assert rtc_flag not in rollout_cmd(ns(**mix), ma, base / "r", "t") and rtc_flag in rollout_cmd(ns(**mix), mb, base / "r", "t")
        assert mb.expected_slow == 0 and ma.expected_slow == 0 and mb.use_peft
        # --n-action-steps 只给同步推理的那个模型
        ma, mb = (make_model("x", p, ns(**{**mix, "n_action_steps": 30}), base) for p in ("pact", "ppi"))
        assert "--policy.n_action_steps=30" in ma.label and "--policy.n_action_steps" not in mb.label

        def refused(case: str, **kw) -> None:
            try:
                run_eval(ns(**kw), Keys(scripted=[]), base)
            except SystemExit as e:
                print(f"{case}，拒绝了：{e}", flush=True)
            else:
                raise AssertionError(f"{case}，却没有拒绝")

        print("\n== 自测 7：几种应该拒绝的情况", flush=True)
        refused("同一个名字换了位置表", **{**mix, "placements": "fixed"})
        refused("--rtc 但两个模型都不是流匹配的", name="x1", policy="pact", name_b="x2", policy_b="pact2", rtc=True)
        refused("只有一个 RTC 模型还给了 --n-action-steps", name="x3", policy="ppi", rtc=True, n_action_steps=30)
        assert len(TRIALS2) == len(TRIALS) == 20 and len(set(TRIALS2)) == 20 and not set(TRIALS2) & set(TRIALS)
        assert all(TRIALS2[i][0] != TRIALS2[i + 1][0] and TRIALS2[i][1] != TRIALS2[i + 1][1] for i in range(19))
        count = lambda lst: sorted(sum(o == k for _, o in lst) for k in ("横放", "竖放", "斜放", "往另一边斜"))  # noqa: E731
        assert count(TRIALS2) == count(TRIALS) and all(p in SPOKEN and o in SPOKEN for p, o in TRIALS2)
        assert len(ROBUST8) == 8 and set(ROBUST8) <= set(TRIALS2)
        assert all(ROBUST8[i][0] != ROBUST8[i + 1][0] and ROBUST8[i][1] != ROBUST8[i + 1][1] for i in range(7))

        print("\n== 自测 8：鲁棒性测试 8 个位置、加卡片的条件", flush=True)
        rb = dict(name="act_rb", policy="pact", name_b="pi_rb", policy_b="ppi", rtc=True, placements="robust8",
                  condition="card")
        run_eval(ns(**rb), Keys(scripted=["start", "success", "start", "fail", "esc"]), base)
        expect("act_rb", [(1, "success")])
        expect("pi_rb", [(1, "fail")])
        assert (rows("act_rb")[0]["position"], rows("act_rb")[0]["orientation"]) == ROBUST8[0]
    print("\n自测全部通过", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", type=str, help="checkpoint 的 pretrained_model 目录")
    ap.add_argument("--name", type=str, help="这一轮评测的名字，用于录像目录和 CSV，例如 act45_100k")
    ap.add_argument("--policy-b", type=str, default=None, help="第二个模型：给了就两个模型交替测")
    ap.add_argument("--name-b", type=str, default=None, help="第二个模型这一轮评测的名字")
    ap.add_argument("--duration", type=float, default=25.0)
    ap.add_argument("--ensemble", type=float, default=None,
                    help="ACT 时间集成系数（如 0.01）：每一步都重新推理并加权平均；不用重新训练")
    ap.add_argument("--n-action-steps", type=int, default=None,
                    help="每个动作块执行几步就重新推理（训练时是 30，即 2 秒）；不用重新训练")
    ap.add_argument("--rtc", action="store_true",
                    help="RTC 异步推理：后台预测、机械臂不停顿，设置见 inference_overrides。"
                         "只加在 π0.5 这类流匹配的模型上，交替测时另一个模型（ACT）照常同步推理")
    ap.add_argument("--placements", choices=sorted(PLACEMENT_SETS), default="fixed",
                    help="用哪份固定位置表：fixed 是第一轮的 20 个（TRIALS），fixed2 是 2026-11-20 定的新 20 个（TRIALS2），"
                         "robust8 是鲁棒性测试的 8 个（ROBUST8，从 TRIALS2 里挑的）")
    ap.add_argument("--condition", choices=list(CONDITIONS), default="clean",
                    help="鲁棒性测试的条件：每次报完位置后语音提醒怎么布置（card 旁边放卡片，lamp 开侧面的灯）；"
                         "只影响语音，条件写进评测名")
    ap.add_argument("--free", type=int, default=None, metavar="N",
                    help="随机摆放 N 次：不用位置表，每次操作者自己换位置和角度（只测一个模型）")
    ap.add_argument("--start", type=int, default=1,
                    help="从第几次开始（只测一个模型时，中途退出后接着测用；交替测时判过的会自动跳过）")
    ap.add_argument("--quiet", action="store_true", help="不出声，只打印")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not (args.policy and args.name):
        ap.error("需要 --policy 和 --name")
    if bool(args.policy_b) != bool(args.name_b):
        ap.error("--policy-b 和 --name-b 要一起给")
    if args.name_b == args.name:
        ap.error("两个模型的名字不能一样，不然结果会写进同一个 CSV")
    if args.ensemble is not None and args.n_action_steps is not None:
        ap.error("--ensemble 和 --n-action-steps 只能选一个")
    if args.free is not None and (args.free < 1 or args.policy_b):
        ap.error("--free 要大于 0，而且只能测一个模型（随机摆的位置没法给第二个模型原样再摆一次）")
    if args.free is not None and args.placements != "fixed":
        ap.error("--free 不用位置表，不能再给 --placements")
    run_eval(args, Keys())


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))  # kill 也走下面的收尾
    try:
        main()
    finally:
        stop_children()
