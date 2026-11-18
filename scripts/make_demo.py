#!/usr/bin/env python3
"""从评测录像剪演示视频（MP4，约 50 秒）和 README 用的动图（GIF）。不用开机械臂。

用的都是评测时存下的录像（data/rollouts/）：
  1. 同一个固定位置，ACT-102（act102_100k_1118）和 π0.5 RTC（pi05_12k_rtc）上下对比，支架相机 + 手臂相机：
     第 6 个位置两个都成功，第 19 个位置只有 π0.5 成功（ACT 合上 3 次都没夹起来）；
  2. π0.5 RTC 随机摆放（pi05_12k_rtc_free）成功的 4 次，2×2；
  3. 随机摆放失败的 2 次：旁边放了干扰物（第 9 次）、送的时候药盒撞到收纳盒前壁掉在外面（第 5 次）；
  4. 结果数字。
动图只有第 2 段里的第 19 个位置，两个模型左右并排。

- 每帧按 frame_times.txt 里的实际时间摆放，画面上的秒数是真实时间，统一 2 倍速。
- "放进收纳盒"的时刻和 eval_stages.py 的判定一样（夹爪在收纳盒上方张开），只标操作者判成功的次数；
  随机第 5 次数据判定也算放进了，其实药盒已经掉在外面，所以失败的次数只写操作者看到的情况。
- 支架相机裁掉房间（床、柜子），只留桌面、从臂和收纳盒；手臂相机只拍到桌面，不裁。
- 用到的片段都逐秒看过，没有人手入镜。ACT 第 15 个位置约 12 秒时操作者伸手把立起来的药盒放平了，所以没用那一段。

    V=~/nexarm-teleop-latency/vendor-nexarm/finsh/.venv/bin/python
    $V scripts/make_demo.py           # MP4 和 GIF 都做
    $V scripts/make_demo.py --gif     # 只做 GIF
输出 outputs/demo/demo.mp4（outputs/ 不进 git）和 docs/img/demo.gif
"""

from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from eval_stages import CLOSE_DELTA, ZONE, frame_times  # noqa: E402

MP4 = ROOT / "outputs" / "demo" / "demo.mp4"
GIF = ROOT / "docs" / "img" / "demo.gif"
ACT, RTC, FREE = "act102_100k_1118", "pi05_12k_rtc", "pi05_12k_rtc_free"

SPEED = 2.0  # 播放倍速
FPS = 30  # MP4 帧率
GIF_FPS = 10
FRONT_CROP = (130, 40, 480, 320)  # 支架相机 640×360 画面里保留的区域 (x, y, 宽, 高)

W, H = 1280, 720
MARGIN, HEAD = 24, 72
BG = (248, 247, 244)
INK = (24, 24, 22)
INK2 = (100, 98, 92)
GOOD = (22, 122, 70)
BAD = (190, 50, 38)
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"


@dataclass
class Trial:
    name: str
    n: int
    result: str  # 操作者判定
    rec: Path
    times: np.ndarray  # 每帧的真实时间（秒）
    closes: list[float]  # 夹爪每次合上的时刻
    release: float | None  # 在收纳盒上方张开的时刻（数据判定）


def load_trial(name: str, n: int) -> Trial:
    """评测 name 里第 n 次（重做过的取最后一次有效的），夹爪事件和 eval_stages.py 的判定一样。"""
    with (ROOT / "outputs" / "eval" / name / "results.csv").open(newline="") as f:
        rows = [r for r in csv.DictReader(f) if int(r["trial"]) == n and r["result"] in ("success", "fail")]
    r = rows[-1]
    rec = ROOT / r["recording"]
    files = sorted(rec.glob("data/chunk-*/file-*.parquet"))
    s = np.stack(pd.concat([pd.read_parquet(p, columns=["observation.state"]) for p in files])["observation.state"].to_numpy())
    g, j1 = s[:, 5], s[:, 0]
    t = frame_times(rec, len(s))
    d = np.diff((g > g[:5].min() + CLOSE_DELTA).astype(int))
    on, off = np.flatnonzero(d == 1) + 1, np.flatnonzero(d == -1) + 1
    rel = [i for i in off if ZONE[0] <= j1[i] <= ZONE[1]]
    return Trial(name, n, r["result"], rec, t, [float(t[i]) for i in on], float(t[rel[0]]) if rel else None)


class Clip:
    """顺序解码一路录像，按真实时间取帧：取时间不晚于 t 的最后一帧。"""

    def __init__(self, tr: Trial, cam: str, size: tuple[int, int]):
        src = tr.rec / "videos" / f"observation.images.{cam}" / "chunk-000" / "file-000.mp4"
        vf = [f"crop={FRONT_CROP[2]}:{FRONT_CROP[3]}:{FRONT_CROP[0]}:{FRONT_CROP[1]}"] if cam == "front" else []
        vf.append(f"scale={size[0]}:{size[1]}:flags=area")
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(src), "-vf", ",".join(vf),
                                      "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.size, self.times, self.i, self.frame = size, tr.times, -1, None

    def _read(self) -> bool:
        w, h = self.size
        buf = self.proc.stdout.read(w * h * 3)
        if len(buf) < w * h * 3:
            return False
        self.frame, self.i = np.frombuffer(buf, np.uint8).reshape(h, w, 3), self.i + 1
        return True

    def at(self, t: float) -> np.ndarray:
        if self.frame is None:
            self._read()
        while self.i + 1 < len(self.times) and self.times[self.i + 1] <= t and self._read():
            pass
        return self.frame

    def close(self) -> None:
        self.proc.stdout.close()
        self.proc.kill()
        self.proc.wait()


@cache
def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT, size)


def text(dr: ImageDraw.ImageDraw, xy, s: str, size: int, color=INK, bold=False, anchor="la") -> None:
    dr.text(xy, s, font=font(size, bold), fill=color, anchor=anchor)


def wrap(dr: ImageDraw.ImageDraw, xy, s: str, size: int, width: int, color=INK2, gap=1.45) -> int:
    """按像素宽度折行写一段英文，返回写完后的 y。"""
    x, y = xy
    line = ""
    for word in s.split():
        trial = f"{line} {word}".strip()
        if dr.textlength(trial, font=font(size)) > width and line:
            text(dr, (x, y), line, size, color)
            y += int(size * gap)
            line = word
        else:
            line = trial
    if line:
        text(dr, (x, y), line, size, color)
        y += int(size * gap)
    return y


def chip(img: Image.Image, xy, s: str, size: int, fg=(255, 255, 255), bg=(0, 0, 0, 150), bold=False) -> None:
    """画面上的半透明小标签。"""
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(over)
    l, t, r, b = od.textbbox(xy, s, font=font(size, bold))
    od.rounded_rectangle((l - 7, t - 5, r + 7, b + 5), 5, fill=bg)
    od.text(xy, s, font=font(size, bold), fill=fg)
    img.paste(Image.alpha_composite(img.convert("RGBA"), over).convert("RGB"))


def status(tr: Trial, t: float, final: bool) -> tuple[str, tuple[int, int, int]]:
    """这一刻的状态：放进收纳盒（只给操作者判成功的）、夹爪合上过几次、结束时没放进。"""
    if tr.result == "success" and tr.release is not None and t >= tr.release:
        return f"In the bin at {tr.release:.1f} s", GOOD
    if final and tr.result == "fail":
        return "Not in the bin by the 40 s limit", BAD
    n = sum(c <= t for c in tr.closes)
    return (f"Gripper closed {n}×" if n else "Reaching for the box"), INK2


def header(dr: ImageDraw.ImageDraw, title: str) -> None:
    text(dr, (MARGIN, 26), title, 28, INK, bold=True)
    text(dr, (W - MARGIN, 30), "2× speed  ·  clocks show real time", 18, INK2, anchor="ra")


def timeline(t_end: float, hold: float, fps: float) -> list[tuple[float, bool]]:
    """每个输出帧对应的真实时间；最后停 hold 秒（标记 final）。"""
    ts = [(float(t), False) for t in np.arange(0, t_end + 1e-9, SPEED / fps)]
    return ts + [(t_end, True)] * int(round(hold * fps))


# ---------- MP4 的各段 ----------

def card(lines: list[tuple[str, int, tuple, bool]], seconds: float):
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    y = 210
    for s, size, color, bold in lines:
        if s:
            text(dr, (W // 2, y), s, size, color, bold, anchor="ma")
        y += int(size * 1.7)
    frame = np.asarray(img)
    return [frame] * int(round(seconds * FPS))


def title_card():
    return card([
        ("ACT vs π0.5 on a desktop robot arm", 44, INK, True),
        ("Task: pick up the medicine box and put it in the bin", 26, INK, False),
        ("", 18, INK, False),
        ("Both policies learned from the same 102 teleoperated demonstrations", 22, INK2, False),
        ("NexArm arm, 5 joints + gripper  ·  front camera + wrist camera  ·  15 Hz control", 22, INK2, False),
        ("ACT: trained on these demos only  ·  π0.5: pretrained VLA, LoRA fine-tuned, real-time chunking (RTC)", 22, INK2, False),
    ], 4.0)


def end_card():
    img = Image.new("RGB", (W, H), BG)
    dr = ImageDraw.Draw(img)
    text(dr, (W // 2, 90), "Results", 40, INK, True, anchor="ma")
    text(dr, (W // 2, 160), "20 fixed placements, one 40 s trial each, same camera position", 22, INK2, anchor="ma")
    cols = (260, 690, 960)
    y = 220
    text(dr, (cols[0], y), "Policy", 20, INK2)
    text(dr, (cols[1], y), "Box in the bin", 20, INK2, anchor="ma")
    text(dr, (cols[2], y), "Median time to release*", 20, INK2, anchor="ma")
    dr.line((cols[0], y + 36, 1090, y + 36), fill=(210, 208, 202), width=2)
    for i, (name, ok, med) in enumerate((("ACT", "12 / 20", "12.2 s"),
                                         ("π0.5, synchronous inference", "17 / 20", "15.3 s"),
                                         ("π0.5, RTC", "17 / 20", "9.9 s"))):
        yy = y + 56 + i * 52
        text(dr, (cols[0], yy), name, 26, INK, i == 2)
        text(dr, (cols[1], yy), ok, 26, INK, i == 2, anchor="ma")
        text(dr, (cols[2], yy), med, 26, INK, i == 2, anchor="ma")
    y = 470
    text(dr, (W // 2, y), "Random placements (π0.5, RTC): 12 / 15 flat placements in the bin,  0 / 2 with distractor objects", 22, INK, anchor="ma")
    text(dr, (W // 2, y + 70), "* over successful trials.  Small samples: ACT vs π0.5 RTC paired sign test p = 0.06.", 18, INK2, anchor="ma")
    text(dr, (W // 2, y + 100), "Flat random placements stayed within the poses covered by the training demonstrations.", 18, INK2, anchor="ma")
    frame = np.asarray(img)
    return [frame] * int(round(7.0 * FPS))


def pair_section(n: int, title: str, t_end: float, hold: float):
    """同一位置，ACT 在上、π0.5 RTC 在下，每行：说明栏 + 支架相机 + 手臂相机。"""
    row_h = (H - HEAD - MARGIN - 16) // 2
    fw, ww = round(row_h * 480 / 320), round(row_h * 640 / 480)
    x_wrist = W - MARGIN - ww
    x_front = x_wrist - 8 - fw
    label_w = x_front - 20 - MARGIN
    rows = []
    for name, policy, sub in ((ACT, "ACT", "Trained on these 102 demos only (ImageNet ResNet18 backbone)"),
                              (RTC, "π0.5", "Pretrained VLA, LoRA fine-tuned on the same 102 demos, RTC")):
        tr = load_trial(name, n)
        rows.append((tr, policy, sub, Clip(tr, "front", (fw, row_h)), Clip(tr, "wrist", (ww, row_h))))
    frames = []
    for t, final in timeline(t_end, hold, FPS):
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img)
        header(dr, title)
        for k, (tr, policy, sub, front, wrist) in enumerate(rows):
            y = HEAD + k * (row_h + 16)
            img.paste(Image.fromarray(front.at(t)), (x_front, y))
            img.paste(Image.fromarray(wrist.at(t)), (x_wrist, y))
            text(dr, (MARGIN, y + 6), policy, 34, INK, True)
            wrap(dr, (MARGIN, y + 58), sub, 18, label_w)
            text(dr, (MARGIN, y + 130), f"{t:4.1f} s", 44, INK, True)
            s, c = status(tr, t, final)
            wrap(dr, (MARGIN, y + 200), s, 22, label_w, c)
        for k in range(2):
            y = HEAD + k * (row_h + 16)
            chip(img, (x_front + 10, y + 8), "front camera", 15)
            chip(img, (x_wrist + 10, y + 8), "wrist camera", 15)
        frames.append(np.asarray(img))
    for _, _, _, a, b in rows:
        a.close()
        b.close()
    return frames


def grid_section(trials: list[int], title: str, side: list[str], notes: dict[int, str], t_end: float, hold: float,
                 rows_n: int):
    """随机摆放的几次：左边说明栏，右边每格一次（支架相机）。"""
    tw, th = 440, 293
    gx = W - MARGIN - (2 * tw + 12)
    gy = HEAD + 8 if rows_n == 2 else HEAD + (H - HEAD - th - 70) // 2
    trs = [load_trial(FREE, n) for n in trials]
    clips = [Clip(tr, "front", (tw, th)) for tr in trs]
    frames = []
    for t, final in timeline(t_end, hold, FPS):
        img = Image.new("RGB", (W, H), BG)
        dr = ImageDraw.Draw(img)
        header(dr, title)
        y = HEAD + 16
        for para in side:
            y = wrap(dr, (MARGIN, y), para, 20, gx - 2 * MARGIN, INK2) + 14
        cells = []
        for i, (tr, clip) in enumerate(zip(trs, clips)):
            x, yy = gx + (i % 2) * (tw + 12), gy + (i // 2) * (th + 12)
            img.paste(Image.fromarray(clip.at(t)), (x, yy))
            cells.append((tr, x, yy))
        for tr, x, yy in cells:
            chip(img, (x + 10, yy + 8), f"trial {tr.n}  ·  {t:4.1f} s", 16)
            if tr.n in notes:
                wrap(dr, (x, yy + th + 12), notes[tr.n], 18, tw, BAD)
                if tr.result == "fail":
                    continue
            s, c = status(tr, t, final)
            chip(img, (x + 10, yy + th - 36), s, 17, bg=(*c, 215) if c != INK2 else (0, 0, 0, 150))
        frames.append(np.asarray(img))
    for c in clips:
        c.close()
    return frames


def make_mp4() -> None:
    sections = [
        title_card,
        lambda: pair_section(6, "Same placement (6 of 20): both succeed", 10.5, 1.5),
        lambda: pair_section(19, "Same placement (19 of 20): only π0.5 succeeds", 20.0, 2.5),
        lambda: grid_section([2, 4, 8, 18], "π0.5 on random placements", [
            "Box placed anywhere on the table at any angle, instead of the 20 fixed test spots.",
            "12 of 15 flat random placements ended in the bin.",
            "In trial 18 the gripper half-closed on the box for several seconds, reopened, and grasped again.",
        ], {}, 20.5, 1.5, 2),
        lambda: grid_section([9, 5], "Where π0.5 failed", [
            "3 of the 15 flat random placements failed, and both trials with distractor objects.",
        ], {9: "A smaller coloured box and a printed card next to it: the gripper never closed (0 / 2 with distractors).",
            5: "Picked up, but the box hit the front wall of the bin and fell outside."}, 16.0, 2.5, 1),
        end_card,
    ]
    MP4.parent.mkdir(parents=True, exist_ok=True)
    enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                            "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "20",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(MP4)], stdin=subprocess.PIPE)
    n = 0
    for make in sections:
        for f in make():
            enc.stdin.write(np.ascontiguousarray(f).tobytes())
            n += 1
    enc.stdin.close()
    enc.wait()
    print(f"{MP4}  {n} 帧，{n / FPS:.1f} 秒，{MP4.stat().st_size / 1e6:.1f} MB")


# ---------- GIF ----------

def make_gif(n: int = 19, t_end: float = 20.0, hold: float = 2.5) -> None:
    """第 n 个位置，ACT 和 π0.5 RTC 左右并排，只用支架相机。"""
    pw, ph, gap, head, foot = 360, 240, 12, 44, 92
    cw = 2 * pw + 3 * gap
    trs = [load_trial(ACT, n), load_trial(RTC, n)]
    clips = [Clip(tr, "front", (pw, ph)) for tr in trs]
    names = ("ACT", "π0.5 (RTC)")
    frames, durations = [], []
    for t, final in timeline(t_end, 0, GIF_FPS) + [(t_end, True)]:
        img = Image.new("RGB", (cw, head + ph + foot), BG)
        dr = ImageDraw.Draw(img)
        for k, (tr, clip) in enumerate(zip(trs, clips)):
            x = gap + k * (pw + gap)
            text(dr, (x + pw // 2, 10), names[k], 22, INK, True, anchor="ma")
            img.paste(Image.fromarray(clip.at(t)), (x, head))
            text(dr, (x, head + ph + 8), f"{t:4.1f} s", 20, INK, True)
            s, c = status(tr, t, final)
            text(dr, (x + 70, head + ph + 12), s, 16, c)
        text(dr, (cw // 2, head + ph + 46), f"Same placement ({n} of 20), 2× speed. Same 102 demos for both.", 15, INK2, anchor="ma")
        text(dr, (cw // 2, head + ph + 68), "All 20 placements within 40 s: ACT 12/20, π0.5 17/20", 15, INK2, anchor="ma")
        frames.append(np.asarray(img))
        durations.append(int(1000 / GIF_FPS))
    durations[-1] = int(hold * 1000)
    for c in clips:
        c.close()
    frames = stabilize(frames)
    # 共用一个调色板：体积小，也不会逐帧闪。画面颜色取 104 种，再加上文字用的颜色（和底色之间的过渡也要，
    # 不然绿色、红色的字被量化成灰的）
    sample = Image.fromarray(np.concatenate(frames[:: max(1, len(frames) // 10)], axis=0))
    colors = np.array(sample.quantize(colors=104, method=Image.Quantize.MEDIANCUT).getpalette()[: 104 * 3]).reshape(-1, 3)
    ui = [np.array(BG) + (np.array(c) - np.array(BG)) * k / 6 for c in (INK, INK2, GOOD, BAD) for k in range(1, 7)]
    palette = Image.new("P", (1, 1))
    palette.putpalette(np.concatenate([colors, np.round(ui), [BG]]).astype(np.uint8).flatten().tolist())
    images = [Image.fromarray(f).quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    GIF.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(GIF, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=True)
    print(f"{GIF}  {len(images)} 帧，{GIF.stat().st_size / 1e6:.1f} MB，播放约 {sum(durations) / 1000:.1f} 秒")


def stabilize(frames: list[np.ndarray], thr: int = 16) -> list[np.ndarray]:
    """和上一帧差得很少的像素沿用上一帧：去掉相机噪声造成的逐帧小变化，GIF 小一半以上。"""
    out, prev = [frames[0]], frames[0].astype(np.int16)
    for f in frames[1:]:
        cur = f.astype(np.int16)
        prev = np.where((np.abs(cur - prev).max(-1) < thr)[..., None], prev, cur)
        out.append(prev.astype(np.uint8))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gif", action="store_true", help="只做 GIF")
    args = ap.parse_args()
    if not args.gif:
        make_mp4()
    make_gif()


if __name__ == "__main__":
    main()
