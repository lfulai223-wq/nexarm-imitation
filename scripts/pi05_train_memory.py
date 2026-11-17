"""π0.5 训练显存实测：和 lerobot-train 一样的一步更新（前向、反向、梯度裁剪、AdamW），合成输入，量峰值显存和每步耗时。

    $V scripts/pi05_train_memory.py <variant> <batch sizes...> [--weights PATH]
    $V scripts/pi05_train_memory.py lora_all 1 4 8 16 --weights ~/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/<版本>/

variant:
  full          全部参数都训练
  freeze_vision 冻结视觉编码器，其余都训练
  expert_only   冻结整个 VLM，只训练动作专家和投影层（LeRobot 的 train_expert_only）
  lora_expert   LoRA，LeRobot 给 π0.5 的默认目标：动作专家的 q/v 投影 + 动作投影层，r=16
  lora_all      LoRA 加在 VLM 和动作专家的所有线性层上，r=16（接近 OpenPI 的 LoRA 微调）

不传 --weights 时随机初始化（显存和预训练权重一样，只是数值不同）；LoRA 必须给 --weights，LeRobot 不让在随机权重上加 LoRA。
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 可以少一点显存碎片。2026-11-17 的结果见 docs/roadmap.md。
输入和我们的数据集同形状：支架相机 360x640、手臂相机 480x640（模型内部缩放到 224x224），语言 + 状态 token 补齐到 200，动作块 50 步。
"""

import argparse
import gc
import time

import torch

ap = argparse.ArgumentParser()
ap.add_argument("variant")
ap.add_argument("batch_sizes", type=int, nargs="+")
ap.add_argument("--weights", default=None)
ap.add_argument("--steps", type=int, default=6)
args = ap.parse_args()

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy

v = args.variant
cfg = PI05Config(dtype="bfloat16", gradient_checkpointing=True, device="cuda",
                 freeze_vision_encoder=(v == "freeze_vision"), train_expert_only=(v == "expert_only"))
cfg.input_features = {
    "observation.images.front": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 360, 640)),
    "observation.images.wrist": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
    "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
}
cfg.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(6,))}

t0 = time.time()
if args.weights:
    cfg.pretrained_path = args.weights
    policy = PI05Policy.from_pretrained(args.weights, config=cfg)
else:
    policy = PI05Policy(cfg)
policy.to("cuda")
print(f"[{v}] 模型就绪 {time.time() - t0:.0f}s；加载后显存 {torch.cuda.memory_allocated() / 2**30:.2f} GiB", flush=True)

if v.startswith("lora"):
    over = {"method_type": "LORA", "r": 16, "target_modules": None, "full_training_modules": None, "init_type": None}
    if v == "lora_all":
        over["target_modules"] = (r"(.*\.(language_model|gemma_expert)\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj"
                                  r"|down_proj)|model\.(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out))")
    policy = policy.wrap_with_peft(peft_cli_overrides=over)

n_all = sum(p.numel() for p in policy.parameters())
trainable = [p for p in policy.parameters() if p.requires_grad]
n_tr = sum(p.numel() for p in trainable)
by_dtype = {}
for p in policy.parameters():
    by_dtype[str(p.dtype)] = by_dtype.get(str(p.dtype), 0) + p.numel()
print(f"[{v}] 参数 {n_all / 1e9:.3f}B，可训练 {n_tr / 1e6:.1f}M；按类型 { {k: f'{n / 1e9:.2f}B' for k, n in by_dtype.items()} }",
      flush=True)

opt = torch.optim.AdamW(trainable, lr=2.5e-5, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.01)
policy.train()


def make_batch(b: int) -> dict:
    mask = torch.zeros(b, 200, dtype=torch.bool, device="cuda")
    mask[:, :60] = True  # 任务文字 + 离散化的状态，大约几十个 token，其余是补齐
    return {
        "observation.images.front": torch.rand(b, 3, 360, 640, device="cuda"),
        "observation.images.wrist": torch.rand(b, 3, 480, 640, device="cuda"),
        "observation.language.tokens": torch.randint(0, 257152, (b, 200), device="cuda"),
        "observation.language.attention_mask": mask,
        "action": torch.randn(b, cfg.chunk_size, 6, device="cuda"),
    }


for b in args.batch_sizes:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        times = []
        for s in range(args.steps):
            batch = make_batch(b)
            torch.cuda.synchronize()
            t = time.perf_counter()
            loss, _ = policy.forward(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
        step = sum(times[2:]) / len(times[2:])
        print(f"[{v}] batch {b:>2}: 峰值显存 {torch.cuda.max_memory_allocated() / 2**30:5.2f} GiB（预留 "
              f"{torch.cuda.max_memory_reserved() / 2**30:5.2f}），每步 {step:.2f} s，loss {loss.item():.3f}", flush=True)
    except torch.OutOfMemoryError:
        print(f"[{v}] batch {b:>2}: 显存不够（OOM），峰值 {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)
        break
