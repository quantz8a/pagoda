#!/usr/bin/env python3
# Validation figures: threshold curves, ROC, score distributions, confusion matrices.
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import font_manager
for _f in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
           "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]:
    try: font_manager.fontManager.addfont(_f)
    except Exception as _e: print("font skip", _f, _e)
plt.rcParams.update({"font.family": ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.dpi": 130})

d = json.load(open("/tmp/umpierre_validation.json"))
rows = d["rows"]
OUT = "/tmp/valfig/"
import os; os.makedirs(OUT, exist_ok=True)
INDIGO, GRAY, GREEN, RED, AMBER = "#5164d6", "#9aa0b4", "#3aa876", "#c0504d", "#d9a441"

def sweep(key):
    ts = np.arange(0.02, 0.99, 0.02)
    sens, spec, keep = [], [], []
    for t in ts:
        tp = sum(1 for r in rows if r["gold"]==1 and (r[key]["p"] or 0) > t)
        fn = sum(1 for r in rows if r["gold"]==1 and not ((r[key]["p"] or 0) > t))
        tn = sum(1 for r in rows if r["gold"]==0 and not ((r[key]["p"] or 0) > t))
        fp = sum(1 for r in rows if r["gold"]==0 and (r[key]["p"] or 0) > t)
        sens.append(tp/(tp+fn)); spec.append(tn/(tn+fp)); keep.append((tp+fp)/len(rows))
    return ts, np.array(sens), np.array(spec), np.array(keep)

ts, sB, pB, kB = sweep("base")
ts, sT, pT, kT = sweep("tuned")

# Fig 1: sensitivity/specificity vs threshold
fig, ax = plt.subplots(figsize=(8.2, 4.6))
ax.plot(ts, sB, color=GRAY, lw=2, label="基座 · 灵敏度")
ax.plot(ts, pB, color=GRAY, lw=2, ls="--", label="基座 · 特异度")
ax.plot(ts, sT, color=INDIGO, lw=2.4, label="微调 · 灵敏度")
ax.plot(ts, pT, color=INDIGO, lw=2.4, ls="--", label="微调 · 特异度")
ax.axhline(0.95, color=RED, lw=1.2, ls=":", label="筛选目标灵敏度 95%")
ax.axvline(0.5, color="#cccccc", lw=1)
ax.scatter([0.36],[0.9706], color=GREEN, zorder=5, s=42)
ax.annotate("基座 t=0.36\n灵敏度 97.1%", (0.36, 0.9706), textcoords="offset points",
            xytext=(10, -26), fontsize=9, color=GREEN)
ax.set_xlabel("纳入判定阈值 t（P(include) > t 则纳入）"); ax.set_ylabel("比例")
ax.set_title("阈值扫描：灵敏度 / 特异度随判定阈值变化（真实 PubMed 金标准, n=279）")
ax.set_ylim(0, 1.03); ax.legend(fontsize=9, loc="center right"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT + "v1-threshold.png"); plt.close(fig)

# Fig 2: ROC-style (1-spec vs sens)
def roc(key):
    pts = [(0.0, 0.0)]
    for t in np.arange(0.99, 0.0, -0.01):
        tp = sum(1 for r in rows if r["gold"]==1 and (r[key]["p"] or 0) > t)
        fn = sum(1 for r in rows if r["gold"]==1 and not ((r[key]["p"] or 0) > t))
        tn = sum(1 for r in rows if r["gold"]==0 and not ((r[key]["p"] or 0) > t))
        fp = sum(1 for r in rows if r["gold"]==0 and (r[key]["p"] or 0) > t)
        pts.append((fp/(fp+tn), tp/(tp+fn)))
    pts.append((1.0, 1.0))
    pts.sort()
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    auc = float(np.trapz(ys, xs))
    return xs, ys, auc
xB, yB, aucB = roc("base"); xT, yT, aucT = roc("tuned")
fig, ax = plt.subplots(figsize=(6.4, 5.4))
ax.plot([0,1],[0,1], color="#cccccc", ls=":", label="随机分类器 (AUC=0.50)")
ax.plot(xB, yB, color=GRAY, lw=2.2, label="基座 Laya (AUC=%.2f)" % aucB)
ax.plot(xT, yT, color=INDIGO, lw=2.2, label="微调 Laya (AUC=%.2f)" % aucT)
ax.set_xlabel("假阳性率（1 - 特异度）"); ax.set_ylabel("真阳性率（灵敏度）")
ax.set_title("ROC：两模型排序能力相当（AUC≈0.74），差异在默认工作点")
ax.legend(fontsize=10, loc="lower right"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(OUT + "v2-roc.png"); plt.close(fig)

# Fig 3: score distributions (violin-ish via boxplot + strip)
fig, ax = plt.subplots(figsize=(8.2, 4.4))
data = [[r["tuned"]["p"] or 0 for r in rows if r["gold"]==1],
        [r["tuned"]["p"] or 0 for r in rows if r["gold"]==0],
        [r["base"]["p"] or 0 for r in rows if r["gold"]==1],
        [r["base"]["p"] or 0 for r in rows if r["gold"]==0]]
labels = ["微调·阳性(n=34)", "微调·阴性(n=245)", "基座·阳性(n=34)", "基座·阴性(n=245)"]
colors = [INDIGO, "#b9c0e8", GRAY, "#d4d6e0"]
bp = ax.boxplot(data, vert=True, patch_artist=True, widths=0.5, showfliers=False, medianprops=dict(color="#1a2340"))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c); patch.set_alpha(0.85)
rng = np.random.default_rng(7)
for i, xs_ in enumerate(data):
    ax.scatter(rng.normal(i+1, 0.045, len(xs_)), xs_, s=5, alpha=0.25, color="#1a2340", zorder=1)
ax.axhline(0.5, color=RED, ls="--", lw=1.2, label="默认阈值 0.5")
ax.set_xticklabels(labels, fontsize=10)
ax.set_ylabel("P(include) 模型输出概率")
ax.set_title("分数分布：微调版把阴性压得很低，但也把一半真实试验压到阈值以下")
ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(OUT + "v3-dist.png"); plt.close(fig)

# Fig 4: confusion matrices at t=0.5
def cm(key):
    tp = sum(1 for r in rows if r["gold"]==1 and r[key]["include"])
    fn = sum(1 for r in rows if r["gold"]==1 and not r[key]["include"])
    tn = sum(1 for r in rows if r["gold"]==0 and not r[key]["include"])
    fp = sum(1 for r in rows if r["gold"]==0 and r[key]["include"])
    return np.array([[tn, fp],[fn, tp]])
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.9))
for ax, key, name in [(axes[0], "base", "基座 Laya"), (axes[1], "tuned", "微调 Laya")]:
    m = cm(key)
    ax.imshow(m, cmap="Blues", vmin=0, vmax=m.max())
    for (i, j), v in np.ndenumerate(m):
        ax.text(j, i, str(v), ha="center", va="center", fontsize=15,
                color="#fff" if v > m.max()*0.55 else "#1a2340", fontweight="bold")
    ax.set_xticks([0,1]); ax.set_xticklabels(["判排除","判纳入"]); ax.set_yticks([0,1]); ax.set_yticklabels(["实际排除","实际纳入"])
    ax.set_title(name + "（阈值 0.5）")
fig.suptitle("混淆矩阵：基座几乎全放行（FN=2），微调版漏掉 18/34 篇真实试验", fontsize=12)
fig.tight_layout(); fig.savefig(OUT + "v4-cm.png"); plt.close(fig)
print("saved 4 figures to", OUT)
print("AUC base=%.3f tuned=%.3f" % (aucB, aucT))
