"""π0.5 推理实测：一次预测一个动作块（10 步去噪）要多久、占多少显存。合成输入，形状和我们的数据一样。

    $V scripts/pi05_infer_latency.py --weights ~/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/<版本>/
"""

import argparse
import statistics
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=None)
ap.add_argument("--n", type=int, default=30)
args = ap.parse_args()

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

cfg = PI05Config(dtype="bfloat16", device="cuda")
cfg.input_features = {
    "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
    "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
}
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}
if args.weights:
    cfg.pretrained_path = args.weights
    policy = PI05Policy.from_pretrained(args.weights, config=cfg)
else:
    policy = PI05Policy(cfg)
policy.to("cuda").eval()

mask = torch.zeros(1, 200, dtype=torch.bool, device="cuda")
mask[:, :60] = True
batch = {
    "observation.images.front": torch.rand(1, 3, 360, 640, device="cuda"),
    "observation.images.wrist": torch.rand(1, 3, 480, 640, device="cuda"),
    "observation.language.tokens": torch.randint(0, 257152, (1, 200), device="cuda"),
    "observation.language.attention_mask": mask,
}
torch.cuda.reset_peak_memory_stats()
times = []
with torch.inference_mode():
    for _ in range(args.n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        chunk = policy.predict_action_chunk(batch)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t)
print(f"动作块 {tuple(chunk.shape)}；第一次 {times[0] * 1000:.0f} ms，之后中位数 {statistics.median(times[3:]) * 1000:.0f} ms"
      f"（最慢 {max(times[3:]) * 1000:.0f}）；峰值显存 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
