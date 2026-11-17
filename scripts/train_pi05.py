#!/usr/bin/env python3
"""从 lerobot/pi05_base 微调 π0.5：就是 lerobot-train，加一处兼容处理。参数和 lerobot-train 完全一样。

    $V scripts/train_pi05.py --policy.type=pi05 --policy.pretrained_path=<pi05_base 目录> ...

为什么要这一层：Hub 上 pi05_base 的前处理配置把"相对动作"这一步叫 relative_actions_processor
（LeRobot 后来改的名字），商家 LeRobot 里同一个类注册的名字是 delta_actions_processor，
直接用 lerobot-train 会在加载前处理配置时报 "not found in registry"。这里给它补注册一个别名。
两边的参数一样（enabled、exclude_joints、action_names），我们不用相对动作（enabled=false），这一步什么都不做。
训练保存的检查点里写的是商家的名字，lerobot-rollout 直接能加载，不需要这个脚本。
"""

import sys

from lerobot.processor.pipeline import ProcessorStepRegistry
from lerobot.processor.relative_action_processor import RelativeActionsProcessorStep

ProcessorStepRegistry._registry.setdefault("relative_actions_processor", RelativeActionsProcessorStep)

from lerobot.scripts.lerobot_train import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
