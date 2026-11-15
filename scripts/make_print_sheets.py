#!/usr/bin/env python3
"""生成项目二要打印的两张 A4：ArUco 码（DICT_4X4_50）和相机标定用的棋盘格。

矢量 PDF，尺寸按毫米精确画。打印时必须选"实际大小 / 100%"，不要选"适合页面"。

    python scripts/make_print_sheets.py
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.font_manager import FontProperties  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "print"  # 生成的 PDF 不进仓库，需要时重新生成
FONT = FontProperties(fname="/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
MM = 1 / 25.4  # 英寸

DICT_NAME = "DICT_4X4_50"
MARKER_MM = 50.0  # 黑色方块（含黑边）的边长
MARKERS = [  # (编号, 用途)
    (0, "任务区左上角"), (1, "任务区右上角"), (2, "任务区右下角"),
    (3, "任务区左下角"), (4, "备用"), (5, "备用"),
]
SQUARE_MM = 25.0  # 棋盘格方格边长
SQUARES = (10, 7)  # 方格数 → 内角点 9 × 6


def page(w_mm: float, h_mm: float):
    fig = plt.figure(figsize=(w_mm * MM, h_mm * MM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w_mm)
    ax.set_ylim(h_mm, 0)  # 原点在左上角，y 向下，单位毫米
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def text(ax, x, y, s, size=9, **kw):
    ax.text(x, y, s, fontproperties=FONT, fontsize=size, va="top", **kw)


def scale_bar(ax, x, y, length_mm=100.0):
    ax.add_patch(Rectangle((x, y), length_mm, 2.0, color="black", lw=0))
    for i in range(int(length_mm) + 1):
        tick = 5.0 if i % 10 == 0 else (3.5 if i % 5 == 0 else 2.0)
        ax.plot([x + i, x + i], [y + 2.0, y + 2.0 + tick], color="black", lw=0.25)
    for i in range(0, int(length_mm) + 1, 10):
        ax.text(x + i, y + 8.0, f"{i}", fontsize=5, ha="center", va="top")
    text(ax, x + length_mm + 4, y - 0.5, f"← 这条刻度尺应正好 {length_mm:.0f} mm", size=8)


def aruco_sheet(path: Path) -> None:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    cells = 6  # 4×4 数据位 + 一圈黑边
    module = MARKER_MM / cells
    quiet = module  # 码外至少留一格宽的白边，剪的时候沿虚线剪

    fig, ax = page(210, 297)
    text(ax, 15, 12, f"ArUco {DICT_NAME}，黑色方块边长 {MARKER_MM:.0f} mm", size=12)
    text(ax, 15, 20, "打印选\"实际大小 / 100%\"，用哑光纸。打印后用尺子量：下面刻度尺应为 100 mm，每个黑色方块应为 50 mm。", size=7.5)
    text(ax, 15, 25, "沿灰色虚线剪下（保留白边），贴平在硬纸板上。码一旦贴好，从录数据到评测都不要再动。", size=7.5)

    cut = MARKER_MM + 2 * quiet  # 剪下来的每一块（含白边）的边长
    col_x = [25 + quiet, 25 + cut + 30 + quiet]  # 两列三行
    row_y = [35 + quiet + k * (cut + 12) for k in range(3)]
    for k, (mid, use) in enumerate(MARKERS):
        x0, y0 = col_x[k % 2], row_y[k // 2]
        img = cv2.aruco.generateImageMarker(dictionary, mid, cells, borderBits=1)
        for r in range(cells):
            for c in range(cells):
                if img[r, c] == 0:
                    ax.add_patch(Rectangle((x0 + c * module, y0 + r * module), module, module, color="black", lw=0))
        ax.add_patch(Rectangle((x0 - quiet, y0 - quiet), MARKER_MM + 2 * quiet, MARKER_MM + 2 * quiet,
                               fill=False, ec="0.6", lw=0.4, ls=(0, (3, 2))))
        text(ax, x0, y0 + MARKER_MM + quiet + 2, f"ID {mid}  {use}", size=8)

    scale_bar(ax, 25, row_y[2] + MARKER_MM + quiet + 12)
    fig.savefig(path)
    plt.close(fig)


def checker_sheet(path: Path) -> None:
    nx, ny = SQUARES
    w, h = nx * SQUARE_MM, ny * SQUARE_MM
    fig, ax = page(297, 210)
    x0, y0 = (297 - w) / 2, (210 - h) / 2
    for r in range(ny):
        for c in range(nx):
            if (r + c) % 2 == 0:
                ax.add_patch(Rectangle((x0 + c * SQUARE_MM, y0 + r * SQUARE_MM), SQUARE_MM, SQUARE_MM, color="black", lw=0))
    text(ax, x0, y0 + h + 4,
         f"标定棋盘格：{nx}×{ny} 格，内角点 {nx - 1}×{ny - 1}，方格 {SQUARE_MM:.0f} mm。"
         f"打印选\"实际大小 / 100%\"，量一下横向 {nx} 格总长应为 {w:.0f} mm。贴在平整硬板上，不能弯。", size=7)
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    aruco_sheet(OUT / "aruco_4x4_50mm_A4.pdf")
    checker_sheet(OUT / "checkerboard_9x6_25mm_A4.pdf")
    print("已生成：", *(str(p) for p in sorted(OUT.glob("*.pdf"))), sep="\n  ")
