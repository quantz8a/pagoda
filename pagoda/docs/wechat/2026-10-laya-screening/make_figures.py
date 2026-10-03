#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""WeChat-article figures for the pagoda x Laya screening story.
Reads results.json (real measured numbers) and renders PNGs into images/.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.font_manager as fm

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


def strip(ax, keep=()):
    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(side in keep)


# ---------------------------------------------------------------- fig1 pipeline
def fig1():
    fig, ax = newfig(10.5, 3.2)
    ax.set_xlim(0, 105); ax.set_ylim(0, 32); ax.axis("off")

    boxes = [
        (2,  "PubMed\n摘要流", GRAY, "40 篇候选文献"),
        (23, "Laya 初筛\n(Rust 单文件)", PRIMARY, "纳入/排除 + 研究设计"),
        (46, "pagoda 网关", INK, "radix 前缀复用"),
        (67, "SGLang 学生\n(蒸馏小模型)", GREEN, "结构化数据抽取"),
        (88, "PRISMA\n报告", ACCENT, "风险偏倚就绪"),
    ]
    W, H, Y = 15, 14, 9
    for x, title, color, sub in boxes:
        ax.add_patch(FancyBboxPatch((x, Y), W, H, boxstyle="round,pad=0.6",
                                    fc=color, ec="none", alpha=0.95))
        ax.text(x + W / 2, Y + H * 0.62, title, ha="center", va="center",
                color="white", fontsize=11.5, fontweight="bold", linespacing=1.3)
        ax.text(x + W / 2, Y - 3.4, sub, ha="center", va="center",
                color=INK, fontsize=9)
    for i in range(len(boxes) - 1):
        x0 = boxes[i][0] + W + 0.6
        x1 = boxes[i + 1][0] - 0.8
        ax.add_patch(FancyArrowPatch((x0, Y + H / 2), (x1, Y + H / 2),
                                     arrowstyle="-|>", mutation_scale=22,
                                     lw=2.2, color=INK))
    fig.savefig(IMG / "fig1-pipeline.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig2 perf
def fig2():
    p = R["laya_perf"]  # keys: latency, cold_start, memory, size (rust, python)
    panels = [
        ("决策延迟 (ms)", p["latency"], 1.0, "越低越好"),
        ("冷启动 (s)", p["cold_start"], 1.0, "越低越好"),
        ("常驻内存 (GB)", p["memory"], 1.0, "越低越好"),
        ("部署体积 (GB)", p["size"], 1.0, "对数刻度"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 3.1), dpi=150)
    fig.patch.set_facecolor(BG)
    for ax, (title, vals, _, note) in zip(axes, panels):
        rust, py = vals
        log = title.startswith("部署体积")
        bars = ax.bar(["Rust", "Python"], [rust, py],
                      color=[PRIMARY, GRAY], width=0.58, log=log)
        ax.set_title(title, fontsize=11, color=INK, pad=8)
        ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center",
                fontsize=8.5, color=GRAY)
        strip(ax, keep=("left",))
        ax.tick_params(axis="y", labelsize=8, colors=GRAY)
        ax.tick_params(axis="x", labelsize=10.5)
        for b, v in zip(bars, [rust, py]):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:g}",
                    ha="center", va="bottom", fontsize=10, color=INK,
                    fontweight="bold")
        ax.margins(y=0.22)
    fig.suptitle("同一个 Laya 模型：Rust 单文件  vs  Python 参考实现",
                 fontsize=13, color=INK, fontweight="bold", y=1.04)
    fig.tight_layout()
    fig.savefig(IMG / "fig2-perf.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig3 accuracy
def fig3():
    a = R["screening"]  # base/tuned x noul/borderline/design + ece
    labels = ["纳入判定", "边界案例", "研究设计分类"]
    keys = ["noul", "borderline", "design"]
    base = [a["base"][k] for k in keys]
    tuned = [a["tuned"][k] for k in keys]

    fig, ax = newfig(7.6, 4.0)
    x = range(len(labels))
    w = 0.34
    b1 = ax.bar([i - w / 2 for i in x], base, w, label="原版 Laya（零样本）",
                color=GRAY)
    b2 = ax.bar([i + w / 2 for i in x], tuned, w,
                label="微调后（LoRA + 温度重标定）", color=PRIMARY)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                    f"{b.get_height():.0%}", ha="center", fontsize=10.5,
                    color=INK, fontweight="bold")
    ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=11.5)
    ax.set_ylim(0, 1.2)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.tick_params(axis="y", labelsize=9, colors=GRAY)
    strip(ax, keep=("left",))
    ax.legend(fontsize=10, frameon=False, loc="lower right")
    ax.set_title(f"文献筛选准确率：{a['n']} 篇种子摘要（含 {a['n_borderline']} 篇刻意设计的边界案例）",
                 fontsize=12, color=INK, pad=10)
    ece = a["ece"]
    ax.text(0.015, 0.965, f"校准误差 ECE：{ece[0]:.3f} → {ece[1]:.3f}（越低越准）",
            transform=ax.transAxes, ha="left", va="top", fontsize=10,
            color=ACCENT, fontweight="bold")
    fig.tight_layout()
    fig.savefig(IMG / "fig3-accuracy.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig4 funnel
def fig4():
    f = R["funnel"]  # total, included, per_item_ms, seconds, train_minutes
    fig, ax = newfig(8.6, 3.6)
    ax.set_xlim(0, 100); ax.set_ylim(0, 34); ax.axis("off")
    rows = [
        ("种子摘要", f["total"], SOFT, INK, 76),
        ("Laya 判定纳入", f["included"], PRIMARY, "white", max(20, 76 * f["included"] / f["total"])),
        ("进入全文精读", f["included"], GREEN, "white", max(20, 76 * f["included"] / f["total"])),
    ]
    H, GAP, y = 8, 3.4, 34 - 8
    for name, n, color, tc, wdt in rows:
        x = (100 - wdt) / 2
        ax.add_patch(FancyBboxPatch((x, y), wdt, H, boxstyle="round,pad=0.45",
                                    fc=color, ec="none"))
        ax.text(50, y + H / 2, f"{name}  {n} 篇", ha="center", va="center",
                color=tc, fontsize=13, fontweight="bold")
        y -= H + GAP
        if y > 0:
            ax.add_patch(FancyArrowPatch((50, y + H + GAP - 0.4), (50, y + H + 0.6),
                                         arrowstyle="-|>", mutation_scale=20,
                                         lw=2, color=GRAY))
    ax.text(50, 1.2, f"单篇判定 {f['per_item_ms']:.0f} ms · {f['total']} 篇全程 {f['seconds']:.1f} s · "
                     f"微调仅 {f['train_minutes']:.0f} 分钟（一张 RTX 3050）",
            ha="center", fontsize=10.5, color=INK)
    fig.savefig(IMG / "fig4-funnel.png", bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- fig0 hero
def fig0():
    fig, ax = newfig(9.0, 4.2)
    ax.set_xlim(0, 90); ax.set_ylim(0, 42); ax.axis("off")
    ax.add_patch(FancyBboxPatch((1, 1), 88, 40, boxstyle="round,pad=0.8",
                                fc=INK, ec="none"))
    # pagoda silhouette: stacked eaves
    cx = 76
    tiers = [(13, 5, 6), (11, 12, 7), (9, 19, 8)]
    for w, y, h in tiers:
        ax.add_patch(FancyBboxPatch((cx - w / 2, y), w, h * 0.55,
                                    boxstyle="round,pad=0.2", fc=ACCENT, ec="none"))
        ax.plot([cx - w / 2 - 2.2, cx + w / 2 + 2.2], [y, y], color=ACCENT,
                lw=3.5, solid_capstyle="round")
    ax.plot([cx, cx], [tiers[-1][1] + tiers[-1][2] * 0.55 + 0.4,
                       tiers[-1][1] + tiers[-1][2] * 0.55 + 4],
            color=ACCENT, lw=3, solid_capstyle="round")
    ax.text(6, 30, "一个 0.4B 的小模型，", color="white", fontsize=23,
            fontweight="bold")
    ax.text(6, 21.5, "替我读完了 40 篇文献", color="white", fontsize=23,
            fontweight="bold")
    ax.text(6, 13.5, "Rust 重写的 Laya · 一张游戏显卡上的系统综述初筛",
            color="#C9D2EE", fontsize=12.5)
    ax.text(6, 8.5, "pagoda · LoRA 微调 · 温度重标定 · radix 前缀复用",
            color=ACCENT, fontsize=10.5)
    fig.savefig(IMG / "fig0-hero.png", bbox_inches="tight", facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    fig0(); fig1(); fig2(); fig3(); fig4()
    print("figures ->", IMG)
