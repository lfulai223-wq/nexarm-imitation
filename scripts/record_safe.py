#!/usr/bin/env python3
"""lerobot-record 加两层保护，参数和 lerobot-record 完全一样，原样传进去。

1. 从臂读数保护，和 scripts/rollout_safe.py 是同一层（做法和原因见那个文件开头）：
   - 连上后要连续两次读数正常而且一致才开始，2 秒内拿不到就报错退出；
   - 录制中读到超范围的值，或者突然跳很多又没被下一次读数确认的值（比如手腕读成 544），就重读，
     几次都不行就沿用上一个好读数。
   商家代码只换掉超出 0–4095 的读数，544 这种值会原样录进数据；遥操作时从臂跟着主臂走，动作不受影响，
   但录下来的状态是坏的。第一批录制时没有出现过，第二批（2026-11-17）起录制都走这个入口。

2. 空的一条不会让程序崩溃（2026-11-17 第二批录到 episode 58 时出过）：
   复位到时间的那一刻正好按了 →（或者保存上一条时多按了一下），这个按键会留到下一条，
   下一条一开始录就结束，一帧都没有；复位结束后保存这条空的，商家代码直接报错退出。
   - 录制开始前就已经收到的 → 忽略（← 和 Esc 照常生效）；
   - 万一还是一帧都没录到，就不保存这条，编号不变，接着录下一条。

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/record_safe.py --config_path=configs/record_formal.yaml --resume=true ...
"""

from __future__ import annotations

import logging

import rollout_safe  # 同一个目录，直接运行本文件时在 sys.path 上

logger = logging.getLogger("record_safe")


def install_record_guards() -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.scripts import lerobot_record

    orig_loop = lerobot_record.record_loop

    def record_loop(*args, **kwargs):
        events = kwargs.get("events")
        # 录制阶段才传 dataset，复位阶段不传
        if kwargs.get("dataset") is not None and events is not None and events.get("exit_early") \
                and not events.get("rerecord_episode") and not events.get("stop_recording"):
            logger.warning("录制开始前就收到了 →（多按了一下，或者正好碰上复位到时间），已忽略，这一条照常录")
            events["exit_early"] = False
        return orig_loop(*args, **kwargs)

    lerobot_record.record_loop = record_loop  # record() 每次按名字找它，换掉模块里的就生效

    orig_save = LeRobotDataset.save_episode

    def save_episode(self, episode_data=None, *args, **kwargs):
        buf = episode_data if episode_data is not None else getattr(self.writer, "episode_buffer", None)
        if isinstance(buf, dict) and buf.get("size", 1) == 0:
            logger.warning("这一条一帧都没录到，不保存，编号不变")
            return None
        return orig_save(self, episode_data, *args, **kwargs)

    LeRobotDataset.save_episode = save_episode


def main() -> None:
    rollout_safe.install()
    install_record_guards()
    from lerobot.scripts import lerobot_record

    lerobot_record.main()


if __name__ == "__main__":
    main()
