#!/usr/bin/env python3
"""lerobot-record 加从臂读数保护，参数和 lerobot-record 完全一样，原样传进去。

保护和 scripts/rollout_safe.py 是同一层（做法和原因见那个文件开头）：
    - 连上后要连续两次读数正常而且一致才开始，2 秒内拿不到就报错退出；
    - 录制中读到超范围的值，或者突然跳很多又没被下一次读数确认的值（比如手腕读成 544），就重读，
      几次都不行就沿用上一个好读数。
商家代码只换掉超出 0–4095 的读数，544 这种值会原样录进数据；遥操作时从臂跟着主臂走，动作不受影响，
但录下来的状态是坏的。第一批录制时没有出现过，第二批（2026-11-17）起录制都走这个入口。

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/record_safe.py --config_path=configs/record_formal.yaml --resume=true ...
"""

from __future__ import annotations

import rollout_safe  # 同一个目录，直接运行本文件时在 sys.path 上


def main() -> None:
    rollout_safe.install()
    from lerobot.scripts import lerobot_record

    lerobot_record.main()


if __name__ == "__main__":
    main()
