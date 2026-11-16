#!/usr/bin/env python3
"""lerobot-rollout 加一层从臂读数保护。参数和 lerobot-rollout 完全一样，原样传进去。

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
        guard = GuardedReads(self.bus.read_positions)
        p = guard.first_good()
        self.bus.read_positions = guard
        self._prev_positions = p  # 商家的 get_observation 用它替换坏值，这里也先填上
        logger.info("从臂读数保护已开启，第一个稳定读数 %s", p)

    NexArmFollower.connect = connect


def main() -> None:
    install()
    from lerobot.scripts import lerobot_rollout

    lerobot_rollout.main()


if __name__ == "__main__":
    main()
