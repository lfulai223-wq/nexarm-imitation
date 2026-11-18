#!/usr/bin/env python3
"""lerobot-rollout 加一层从臂读数保护。参数和 lerobot-rollout 完全一样，原样传进去。

另外三处（2026-11-18 为 π0.5 加的，ACT 不受影响，见各函数说明）：
π0.5 建模型时跳过随机初始化（加载 76 秒 → 12 秒）；LoRA 加载后合并进权重（重新预测 373 → 245 ms）；
录像时记下每一帧的真实时间（frame_times.txt，控制循环超时不补帧，按帧数算时间会偏短）。

为什么要有这个（2026-11-16 开评测前发现）：
    从臂连上后的第一次读数偶尔是坏的：关节 4–6 读成 4095 / 544 / 4095，就是舵机链末端没应答的那种读错。
    商家代码只在"已经有上一次读数"时才把坏值换掉，第一次读数没有上一次，坏值原样通过。
    lerobot-rollout 恰好把这第一次读数存成"初始姿势"，结束时用 3 秒把从臂插值回去，
    手腕和夹爪会被拧到行程尽头；策略第一次推理看到的也是这个坏状态。
    昨天 19:41 和今天 11:27 两次遥操作连接时都出现过（遥操作只跟主臂走，没受影响）。

做法：
    - 连上以后连续读，直到两次读数都在正常范围、而且彼此一致，才算拿到第一个好读数；
      2 秒内拿不到就报错退出（评测工具会记成 error），从臂不动。
    - 之后每次读数：超出范围（≤5 或 ≥4090，和商家代码的判坏阈值一样），或者和上一个好读数相差超过
      300 计数又没被紧接着的下一次读数确认，就重读；重读 4 次还不行就沿用上一个好读数，并打警告。
      从臂速度上限 2000 计数/秒，15 Hz 下一步最多走约 133 计数，正常运动不会触发。

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/rollout_safe.py --config_path=configs/rollout_base.yaml --policy.path=...
"""

from __future__ import annotations

import logging
import time

LOW, HIGH = 5, 4090  # 读数 ≤ LOW 或 ≥ HIGH 当作坏值
MAX_JUMP = 300       # 和上一个好读数相差超过这么多，要再读一次确认
AGREE = 50           # 两次读数相差在这以内算一致
RETRIES = 4
FIRST_READ_TIMEOUT_S = 2.0
# 串口写超时：商家驱动是 50 ms。pyserial 写完几个字节后会检查是否已超时；RTC 后台线程推理时会占住 Python 全局锁，
# 主线程写完一拿回锁就可能已过 50 ms，于是报 SerialTimeoutException（写其实已经成功）。2026-11-18 RTC 评测前 11 次里 4 次，
# 都在控制循环开始的第一秒，预热模型后仍有。放宽到 0.5 秒只影响这个检查；读超时不变，同步推理没有这个问题。
WRITE_TIMEOUT_S = 0.5

logger = logging.getLogger("rollout_safe")


def _in_range(p: list[int]) -> bool:
    return all(LOW < x < HIGH for x in p)


def _diff(a: list[int], b: list[int]) -> int:
    return max(abs(x - y) for x, y in zip(a, b))


class GuardedReads:
    """包住总线的 read_positions：只把可信的读数交出去。"""

    def __init__(self, raw_read, sleep=time.sleep, clock=time.perf_counter):
        self.raw_read = raw_read
        self.sleep = sleep
        self.clock = clock
        self.good: list[int] | None = None
        self.rejected = 0

    def first_good(self) -> list[int]:
        deadline = self.clock() + FIRST_READ_TIMEOUT_S
        prev, p, n = None, None, 0
        while self.clock() < deadline:
            try:
                p = list(self.raw_read())
            except TimeoutError:
                prev = None
                continue
            n += 1
            if _in_range(p) and prev is not None and _diff(p, prev) <= AGREE:
                self.good = p
                if n > 2:
                    logger.warning("从臂连上后前 %d 次读数里有坏的，第一个稳定读数 %s", n - 2, p)
                return p
            prev = p if _in_range(p) else None
            self.sleep(0.02)
        raise RuntimeError(f"从臂连上后 {FIRST_READ_TIMEOUT_S:.0f} 秒内读不到稳定的位置（最后一次 {p}），不启动")

    def __call__(self, *args, **kwargs) -> list[int]:
        assert self.good is not None, "先调用 first_good()"
        last, p = None, None
        for _ in range(RETRIES):
            p = list(self.raw_read(*args, **kwargs))
            if _in_range(p):
                if _diff(p, self.good) <= MAX_JUMP:
                    self.good = p
                    return p
                if last is not None and _diff(p, last) <= AGREE:  # 连续两次都这样：是真的动了
                    self.good = p
                    return p
                last = p
            self.rejected += 1
        logger.warning("从臂读数连续 %d 次异常（最后一次 %s），沿用上一个好读数 %s", RETRIES, p, self.good)
        return list(self.good)


def install() -> None:
    from lerobot.robots.nexarm_follower.nexarm_follower import NexArmFollower

    orig_connect = NexArmFollower.connect

    def connect(self, *args, **kwargs):
        orig_connect(self, *args, **kwargs)
        self.bus._serial.write_timeout = WRITE_TIMEOUT_S
        guard = GuardedReads(self.bus.read_positions)
        p = guard.first_good()
        self.bus.read_positions = guard
        self._prev_positions = p  # 商家的 get_observation 用它替换坏值，这里也先填上
        logger.info("从臂读数保护已开启，第一个稳定读数 %s", p)

    NexArmFollower.connect = connect


MERGE_MARK = "LoRA 已合并进权重"  # eval_trials.py 对 LoRA 微调的模型检查这一行


def install_pi05_fast_load() -> None:
    """π0.5 建模型时跳过随机初始化：41 亿个参数随机初始化要 73 秒，反正马上被检查点覆盖。
    2026-11-18 实测：加载 76 秒 → 12 秒；跳过和不跳过加载出来的 state_dict 813 项完全一样，
    同一噪声下的动作块也完全一样，参数和缓冲没有 NaN/inf。只影响 π0.5，ACT 不经过这里。"""
    from transformers.initialization import no_init_weights

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    orig = PI05Policy.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        with no_init_weights():
            policy = orig(cls, *args, **kwargs)
        _warmup_pi05(policy)
        return policy

    PI05Policy.from_pretrained = classmethod(from_pretrained)


def _warmup_pi05(policy) -> None:
    """连机械臂之前先用假输入推理两次，把第一次推理的冷启动（CUDA 内核加载、cuBLAS 初始化等）做掉。

    为什么（2026-11-18 RTC 评测时发现）：RTC 在后台线程推理，主循环同时在读写从臂串口。第一次推理冷启动时
    长时间占住 Python 的全局锁，主循环拿不到锁；商家驱动串口写超时只有 50 ms，pyserial 写完一检查已经超时，
    报 SerialTimeoutException。前 8 次里出现 3 次，都在控制循环开始的第一秒。同步推理时推理和串口在同一个线程，没有这个问题。
    顺便把求导引擎也启动一下（RTC 的引导项用 torch.autograd.grad）。"""
    import torch

    from lerobot.configs.types import FeatureType

    dev = next(policy.parameters()).device
    if dev.type != "cuda":
        return
    cfg = policy.config
    batch = {k: (torch.rand if ft.type is FeatureType.VISUAL else torch.zeros)(1, *ft.shape, device=dev)
             for k, ft in cfg.input_features.items() if ft.type in (FeatureType.VISUAL, FeatureType.STATE)}
    n = cfg.tokenizer_max_length
    mask = torch.zeros(1, n, dtype=torch.bool, device=dev)
    mask[:, :30] = True
    batch["observation.language.tokens"] = torch.zeros(1, n, dtype=torch.long, device=dev)
    batch["observation.language.attention_mask"] = mask
    t = time.perf_counter()
    with torch.no_grad():
        for _ in range(2):
            policy.predict_action_chunk(batch)
    x = torch.zeros(1, device=dev, requires_grad=True)
    torch.autograd.grad((x * 2).sum(), x)
    torch.cuda.synchronize()
    logger.info("π0.5 预热完成（%.1f 秒）", time.perf_counter() - t)


def install_lora_merge() -> None:
    """LoRA 微调的检查点（π0.5）：lerobot-rollout 加载后不合并，每层推理都多算一条 LoRA 支路。
    2026-11-18 离线实测（batch 1，含前后处理）：重新预测一个动作块 373 ms → 合并后 245 ms，普通一步 6.9 → 2.8 ms；
    同一噪声下合并前后动作块最大差 0.0074（归一化单位，约 1 个计数）。所以加载完就合并。"""
    from peft import PeftModel

    orig = PeftModel.from_pretrained

    def from_pretrained(*args, **kwargs):
        merged = orig(*args, **kwargs).merge_and_unload()
        logger.info(MERGE_MARK)
        return merged

    PeftModel.from_pretrained = from_pretrained


def install_frame_clock(argv: list[str]) -> None:
    """记下每个录下来的帧对应的真实时间，存到录像目录的 frame_times.txt（每行一个秒数，从第一帧算起）。

    为什么：控制循环超时的那一步不会补帧，所以录像的帧数比 时长 × 15 少，按"第几帧 / 15"算时间会偏短。
    ACT 每次只少几帧；π0.5 每次重新预测要停约 0.25 秒，40 秒少约 70 帧，按帧数算的"25 秒内"会多给它约 3 秒。
    eval_stages.py 有这个文件时用它算时间。时间取每一步拿到观测、开始算动作的时刻（就是这一帧画面和读数的时刻）。"""
    root = next((a.split("=", 1)[1] for a in argv if a.startswith("--dataset.root=")), None)
    if root is None:
        return
    import atexit
    from pathlib import Path

    import lerobot.rollout.strategies.sentry as sentry

    times: list[float] = []
    orig = sentry.send_next_action

    def send_next_action(*args, **kwargs):
        t = time.perf_counter()
        out = orig(*args, **kwargs)
        if out is not None:  # 只有拿到动作的那一步才会录一帧
            times.append(t)
        return out

    sentry.send_next_action = send_next_action

    def write() -> None:
        if times and Path(root).is_dir():
            Path(root, "frame_times.txt").write_text("".join(f"{t - times[0]:.4f}\n" for t in times))

    atexit.register(write)


def main() -> None:
    import sys

    install()
    install_pi05_fast_load()
    install_lora_merge()
    install_frame_clock(sys.argv[1:])
    from lerobot.scripts import lerobot_rollout

    lerobot_rollout.main()


if __name__ == "__main__":
    main()
