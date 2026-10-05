#!/usr/bin/env python3
# Round-2 figures: v1-base / v1-tuned / v2-tuned comparison.
import json
import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    try: font_manager.fontManager.addfont(_f)
    except Exception: pass
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 130})

v1 = json.load(open("/tmp/umpierre_validation.json"))["rows"]
v2 = json.load(open("/tmp/umpierre_validation_v2.json"))
OUT = "/tmp/valfig/"
INDIGO, GRAY, GREEN, RED = "#5164d6", "#9aa0b4", "#3aa876", "#c0504d"

series = [("基座（零样本）", [("base", r) for r in v1], GRAY),
          ("微调 v1（合成数据）", [("tuned", r) for r in v1], INDIGO),
          ("微调 v2（真实分布+FN加权）", [("v2", r) for r in v2], GREEN)]

def getp(item):
    kind, r = item
    return (r["p"] if kind == "v2" else (r[kind]["p"])) or 0

# Fig R2-1: threshold sweep, 3 models
fig, ax = plt.subplots(figsize=(8.6, 4.8))
for name, data, color in series:
    ts = np.arange(0.02, 0.99, 0.02)
    sens = []
    for t in ts:
        tp = sum(1 for it in data if it[1]["gold"]==1 and getp(it) > t)
        fn = sum(1 for it in data if it[1]["gold"]==1 and getp(it) <= t)
        sens.append(tp/(tp+fn))
    ax.plot(ts, sens, color=color, lw=2.4, label=name)
ax.axhline(0.95, color=RED, lw=1.2, ls=":", label="初筛目标灵敏度 95%")
ax.axvline(0.5, color="#cccccc", lw=1)
ax.scatter([0.5],[1.0], color=GREEN, s=48, zorder=5)
ax.annotate("v2 @ t=0.5：灵敏度 100%", (0.5, 1.0), textcoords="offset points",
            xytext=(-150, -14), fontsize=9, color=GREEN, fontweight="bold")
ax.scatter([0.5],[0.4706], color=INDIGO, s=48, zorder=5, marker="v")
ax.annotate("v1 @ t=0.5：47.1%", (0.5, 0.4706), textcoords="offset points",
            xytext=(8, -4), fontsize=9, color=INDIGO)
ax.set_xlabel("纳入判定阈值 t（P(include) > t 则纳入）"); ax.set_ylabel("灵敏度（不漏诊率）")
ax.set_title("补考成绩：v2 在 t ≤ 0.56 全区间灵敏度 100%（n=279 真实摘要）\n注：v1 需 t<0.01 极低阈值才能达 97%，但此时特异度仅 27%，实际不可用")
ax.set_ylim(0, 1.05); ax.legend(fontsize=9, loc="lower left"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT + "v5-round2-threshold.png"); plt.close(fig)

# Fig R2-2: summary bars at each model's best sens>=0.95 operating point
fig, ax = plt.subplots(figsize=(8.2, 4.2))
names = ["基座 t=0.36", "v1 微调 t=0.0085", "v2 微调 t=0.68", "v2 微调 t=0.5（默认）"]
sens_v = [0.971, 0.971, 0.971, 1.000]
spec_v = [0.200, 0.274, 0.478, 0.449]
x = np.arange(len(names)); w = 0.36
b1 = ax.bar(x - w/2, [s*100 for s in sens_v], w, color=GREEN, label="灵敏度")
b2 = ax.bar(x + w/2, [s*100 for s in spec_v], w, color=GRAY, label="特异度（=省掉的人力）")
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+1.5, "%.0f" % b.get_height(),
                ha="center", fontsize=9, fontweight="bold")
ax.axhline(95, color=RED, ls=":", lw=1.2)
ax.text(3.42, 96, "95%", color=RED, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9.5)
ax.set_ylabel("%"); ax.set_ylim(0, 112)
ax.set_title("灵敏度≥97% 前提下的省人力对比：v2 特异度 47.8%，是基座的 2.4 倍")
ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(OUT + "v6-round2-bars.png"); plt.close(fig)
print("saved v5/v6")
