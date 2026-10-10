#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Figures for the hot-scenarios WeChat article. Numbers from results.json (measured)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = Path(__file__).parent
IMG = HERE / "images"
IMG.mkdir(exist_ok=True)
R = json.loads((HERE / "results.json").read_text(encoding="utf-8"))

INK = "#2B3452"; PRIMARY = "#4F6BD8"; ACCENT = "#E8A33D"
GREEN = "#3FA97A"; RED = "#D8574F"; GRAY = "#9AA0AC"; BG = "#FFFFFF"
SOFT = "#EEF1F8"


def newfig(w, h):
    fig, ax = plt.subplots(figsize=(w, h), dpi=150)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    return fig, ax


def rbox(ax, x, y, w, h, color, alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.2",
                                fc=color, ec="none", alpha=alpha))


# ---------------------------------------------------------- fig0 hero
def fig0():
    fig, ax = newfig(9.0, 4.6)
    fig.patch.set_facecolor(INK)
    ax.set_facecolor(INK)
    ax.set_xlim(0, 90); ax.set_ylim(0, 46); ax.axis("off")

    # top pill badge
    rbox(ax, 27, 39, 36, 4.6, "none")
    ax.add_patch(FancyBboxPatch((27, 39), 36, 4.6, boxstyle="round,pad=0.6,rounding_size=2.3",
                                fc="none", ec=ACCENT, lw=1.4))
    ax.text(45, 41.3, "GitHub 17K Star · System-1 决策模型", fontsize=10.5,
            color=ACCENT, ha="center", va="center", fontweight="bold")

    # title: 不聊天， muted; 只拍板 bright + accent underline bar
    ax.text(29.8, 30.5, "不聊天，", fontsize=30, color="#C9CFDE", ha="left", fontweight="bold")
    ax.text(46.9, 30.5, "只拍板", fontsize=30, color=ACCENT, ha="left", fontweight="bold")

    ax.text(45, 22.5, "Rust 复刻 Laya 的 4 个最火玩法", fontsize=13.5,
            color="#C9CFDE", ha="center")

    # four cards in ONE neat row
    cards = [
        ("重复扣费?", "billing 0.99", PRIMARY),
        ("擦边内容?", "fraud 0.80", ACCENT),
        ("生产事故?", "escalate", RED),
        ("耳机没声音?", "refund 1.00", GREEN),
    ]
    for i, (q, a, c) in enumerate(cards):
        x = 4 + i * 21.2
        rbox(ax, x, 6.5, 19, 9.5, "#39415E")
        ax.text(x + 9.5, 12.2, q, fontsize=9.5, color="#C9CFDE", ha="center")
        ax.text(x + 9.5, 8.6, a, fontsize=11.5, color=c, ha="center", fontweight="bold")

    ax.text(45, 2.2, "pagoda · SGLang 架构思想的 Rust 重实现", fontsize=9,
            color="#8A90A8", ha="center")
    fig.savefig(IMG / "fig0-hero.png", bbox_inches="tight", facecolor=INK)
    plt.close(fig)

# ---------------------------------------------------------- fig1 scenarios
def fig1():
    fig, ax = newfig(10.5, 5.6)
    ax.set_xlim(0, 105); ax.set_ylim(0, 56); ax.axis("off")

    cards = [
        (2,  29, "玩法 1 · 客服工单路由", "输入：重复扣费，今天不退款就取消",
         ["department = billing (0.99)", "churn_risk = 是, P=0.88"], PRIMARY),
        (54, 29, "玩法 2 · 内容安全护栏", "输入：帮我破解邻居的 WiFi",
         ["is_harmful P=0.27 (看着不高)", "category = fraud (0.80) 命中"], ACCENT),
        (2,  3,  "玩法 3 · Agent 判断器", "输入：你们把我生产库删了！",
         ["need_human P=0.68", "next_action = escalate"], RED),
        (54, 3,  "玩法 4 · 中文意图识别", "输入：耳机左耳没声音，申请退款",
         ["intent = refund (1.00)", "angry = 否, P=0.008"], GREEN),
    ]
    for x, y, title, inp, outs, c in cards:
        rbox(ax, x, y, 49, 24, SOFT)
        rbox(ax, x, y + 18.5, 49, 5.5, c)
        ax.text(x + 24.5, y + 21.2, title, fontsize=12, color="white",
                ha="center", va="center", fontweight="bold")
        ax.text(x + 2.5, y + 14.6, inp, fontsize=9.5, color="#5A6272")
        for i, o in enumerate(outs):
            ax.text(x + 2.5, y + 9.6 - i * 4.6, o, fontsize=10.5,
                    color=INK, fontweight="bold")
    fig.savefig(IMG / "fig1-scenarios.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- fig2 latency
def fig2():
    fig, ax = newfig(9.5, 4.2)
    rows = [
        ("pagoda Rust · GPU (f32)", R["gpu_p50_ms"], PRIMARY),
        ("官方 Python · GPU (fp16)", R["py_gpu_ms"], GRAY),
        ("pagoda Rust · CPU (f32)", R["cpu_ms"], "#8FA3E8"),
        ("多语言检查点 · GPU", R["multilingual_ms"], GREEN),
    ]
    labels = [r[0] for r in rows][::-1]
    vals = [r[1] for r in rows][::-1]
    colors = [r[2] for r in rows][::-1]
    bars = ax.barh(labels, vals, color=colors, height=0.58)
    for b, v in zip(bars, vals):
        ax.text(v + 12, b.get_y() + b.get_height() / 2, f"{v:.0f} ms",
                va="center", fontsize=11, color=INK, fontweight="bold")
    ax.set_xlim(0, 1300)
    ax.set_xlabel("单次决策延迟（ms，越低越好）· RTX 3050 8GB 实测", fontsize=10)
    ax.tick_params(axis="y", labelsize=10.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.text(0.99, 0.04, "注：官方标称 ~33ms 为高端卡 + fp16 口径",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=GRAY)
    fig.tight_layout()
    fig.savefig(IMG / "fig2-latency.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- fig3 cublas
def fig3():
    fig, ax = newfig(10.5, 3.6)
    ax.set_xlim(0, 105); ax.set_ylim(0, 36); ax.axis("off")

    steps = [("1 建设备", "OK", GREEN), ("2 拷显存", "OK", GREEN),
             ("3 矩阵乘", "FAIL", RED), ("4 reduce", "—", GRAY)]
    x = 2
    for name, st, c in steps:
        rbox(ax, x, 22, 20, 10, SOFT)
        ax.text(x + 10, 28.8, name, fontsize=11, ha="center", color=INK, fontweight="bold")
        ax.text(x + 10, 24.6, st, fontsize=11, ha="center", color=c, fontweight="bold")
        x += 25
    ax.text(2, 34.5, "cuda_smoke 四步定位法", fontsize=12, color=INK, fontweight="bold")

    ar = FancyArrowPatch((50, 20.5), (50, 15.5), arrowstyle="-|>",
                         mutation_scale=16, color=GRAY)
    ax.add_patch(ar)
    rbox(ax, 2, 4.0, 49, 12, "#FBEEEA")
    ax.text(4.5, 12.6, "根因", fontsize=11, color=RED, fontweight="bold")
    ax.text(4.5, 9.2, "libcublas.so.12 被解析成 12.8（ollama 自带）", fontsize=9, color=INK)
    ax.text(4.5, 6.0, "驱动却是 12.2 时代（535）→ GEMM 全挂", fontsize=9, color=INK)
    rbox(ax, 54, 4.0, 49, 12, "#EAF5EF")
    ax.text(56.5, 12.6, "修复", fontsize=11, color=GREEN, fontweight="bold")
    ax.text(56.5, 9.2, "LD_LIBRARY_PATH 前置与驱动同代的", fontsize=9, color=INK)
    ax.text(56.5, 6.0, "cuBLAS 12.2 → 立刻痊愈", fontsize=9, color=INK)
    fig.savefig(IMG / "fig3-cublas.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------- fig4 deploy
def fig4():
    fig, ax = newfig(9.5, 4.4)
    ax.set_xlim(0, 95); ax.set_ylim(0, 44); ax.axis("off")

    ax.text(22, 41, "官方 Python 栈", fontsize=13, ha="center", color=GRAY, fontweight="bold")
    stack = ["Python 环境", "PyTorch（数 GB）", "transformers", "CUDA 工具链", "冷启动按分钟计"]
    for i, s in enumerate(stack):
        y = 33 - i * 7
        rbox(ax, 4, y, 36, 5.2, SOFT)
        ax.text(22, y + 2.6, s, fontsize=10.5, ha="center", va="center", color="#5A6272")

    ax.text(70, 41, "pagoda Rust 版", fontsize=13, ha="center", color=PRIMARY, fontweight="bold")
    rbox(ax, 48, 12, 44, 26, PRIMARY)
    ax.text(70, 31.5, "单个二进制文件", fontsize=14, ha="center", color="white", fontweight="bold")
    ax.text(70, 25.5, "bash scripts/serve-laya.sh 一键起服", fontsize=10.5, ha="center", color="#DDE3F5")
    ax.text(70, 20.5, "HTTP API 与 Jev 兼容：改个 URL 即切换自托管", fontsize=10.5, ha="center", color="#DDE3F5")
    ax.text(70, 15.5, "无 GC 停顿 · 无 Python 依赖", fontsize=10.5, ha="center", color="#DDE3F5")

    ar = FancyArrowPatch((43, 25), (51, 25), arrowstyle="-|>", mutation_scale=22, color=ACCENT)
    ax.add_patch(ar)
    fig.savefig(IMG / "fig4-deploy.png", bbox_inches="tight")
    plt.close(fig)


fig0(); fig1(); fig2(); fig3(); fig4()
print("figures ->", IMG)
