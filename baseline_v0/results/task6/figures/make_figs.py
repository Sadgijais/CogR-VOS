import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

# palette (validated: slots 1-2 of the reference categorical palette, light mode)
BLUE, ORANGE = "#2a78d6", "#eb6834"      # MeViS, Long-RVOS (same in every figure)
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, SURF, BAND = "#e6e5e1", "#fcfcfb", "#efeeea"
DS = {"MeViS": BLUE, "Long-RVOS": ORANGE}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "text.color": INK,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1, "axes.axisbelow": True,
})
NOTE = "Selected subsets, single runs; not official benchmark results. Sentences of one video are not independent."

def save(fig, name):
    fig.savefig(name + ".png", dpi=200, bbox_inches="tight")
    fig.savefig(name + ".pdf", bbox_inches="tight")
    plt.close(fig)

def foot(fig, text, y=-0.02):
    fig.text(0.01, y, text, fontsize=8, color=MUTED, ha="left", va="top", wrap=True)

# ---------- Figure 1: top-k candidate recall ----------
k = np.arange(1, 6)
mevis = [37.7, 68.1, 77.5, 81.2, 82.7]
lrvos = [27.6, 48.3, 62.1, 75.9, 86.2]
fig, ax = plt.subplots(figsize=(7.2, 4.3))
for name, ys in (("MeViS (n = 191)", mevis), ("Long-RVOS (n = 29)", lrvos)):
    c = BLUE if name.startswith("MeViS") else ORANGE
    ax.plot(k, ys, color=c, lw=2, marker="o", ms=7, mec=SURF, mew=2, label=name, solid_capstyle="round")
    up = name.startswith("MeViS") if True else False
    last_up = not name.startswith("MeViS")   # Long-RVOS ends higher at k=5
    for xi, yi, above in ((1, ys[0], up), (5, ys[-1], last_up)):
        ax.text(xi, yi + (5.5 if above else -5.5), f"{yi:.1f}%", ha="center", va="center", fontsize=9, color=INK2)
ax.text(2.0, 72.6, "68.1%", fontsize=9, color=INK2, ha="center")
ax.set_xticks(k); ax.set_xticklabels(["top-1\n(current)", "top-2", "top-3", "top-4", "top-5\n(whole pool)"])
ax.set_ylim(0, 100); ax.set_yticks(range(0, 101, 20)); ax.set_yticklabels([f"{v}%" for v in range(0, 101, 20)])
ax.set_xlim(0.6, 5.4); ax.grid(axis="x", visible=False)
ax.set_ylabel("Sentences whose target is among the top-k boxes")
ax.set_title("The right box is usually in the pool, but rarely ranked first", loc="left", fontsize=11, fontweight="bold", color=INK)
ax.legend(frameon=False, loc="lower right", labelcolor=INK2)
foot(fig, "Sentences with the target visible at frame 0 (hit = box IoU >= 0.5 with ground truth).\n33 of 191 (MeViS) and 4 of 29 (Long-RVOS) have the target outside the pool.\nSelected subsets, single runs; not official benchmark results.\nSentences of one video are not independent.", y=-0.01)
save(fig, "fig1_topk_recall")

# ---------- Figure 2: selection arms ----------
arms = ["A0\ntop-1", "A1\nCLIP", "A3-5", "A3-10", "Largest\nbox"]
jf = {"MeViS": [26.88, 27.06, 32.16, 31.51, 28.16], "Long-RVOS": [30.38, 37.77, 35.12, 33.58, 26.85]}
oracle = {"MeViS": (52.80, 64.29), "Long-RVOS": (61.22, 61.15)}   # best cached candidate, GT box
diff = {"MeViS": [(0.19, -4.43, 6.10), (5.28, -0.74, 12.21), (4.63, -2.49, 12.54), (1.28, -3.31, 8.12)],
        "Long-RVOS": [(7.39, -3.58, 20.19), (4.73, -2.70, 12.05), (3.20, -4.40, 10.90), (-3.53, -23.33, 17.14)]}
fig, axs = plt.subplots(2, 2, figsize=(11, 7.6), gridspec_kw={"height_ratios": [1.15, 1]})
for j, ds in enumerate(("MeViS", "Long-RVOS")):
    c = DS[ds]; ax = axs[0, j]
    x = np.arange(5)
    ax.bar(x, jf[ds], width=0.5, color=c, zorder=3)
    for xi, v in zip(x, jf[ds]):
        ax.text(xi, v + 1.0, f"{v:.1f}", ha="center", va="bottom", fontsize=9, color=INK)
    cand, gt = oracle[ds]
    ax.axhline(cand, color=MUTED, lw=1.2, ls=(0, (5, 3)), zorder=2)
    ax.axhline(gt, color=INK2, lw=1.2, ls=(0, (1, 2)), zorder=2)
    if ds == "MeViS":
        ax.text(4.45, cand - 1.0, f"best cached candidate start {cand:.1f}", ha="right", va="top", fontsize=8, color=INK2)
        ax.text(4.45, gt + 0.8, f"ground-truth box start {gt:.1f}", ha="right", va="bottom", fontsize=8, color=INK2)
    else:
        ax.text(4.45, cand + 0.8, f"best cached candidate {cand:.1f} / ground-truth box {gt:.1f}", ha="right", va="bottom", fontsize=8, color=INK2)
    ax.set_xticks(x); ax.set_xticklabels(arms); ax.set_ylim(0, 72); ax.grid(axis="x", visible=False)
    ax.set_title(ds, loc="left", fontsize=11, fontweight="bold", color=INK)
    if j == 0: ax.set_ylabel("J&F")
    ax = axs[1, j]
    ax.axvline(0, color=INK2, lw=1, zorder=2)
    names = ["A1 CLIP", "A3-5 look-ahead", "A3-10 look-ahead", "Largest box"]
    for i, (d, lo, hi) in enumerate(diff[ds]):
        y = 3 - i
        ax.plot([lo, hi], [y, y], color=c, lw=2, solid_capstyle="round", zorder=3)
        ax.plot([d], [y], "o", color=c, ms=8, mec=SURF, mew=2, zorder=4)
        ax.text(hi + 0.8, y, f"{d:+.1f}  [{lo:+.1f}, {hi:+.1f}]", va="center", fontsize=8.5, color=INK2)
    ax.set_yticks([3, 2, 1, 0]); ax.set_yticklabels(names); ax.set_ylim(-0.7, 3.7)
    ax.set_xlim(-26 if ds == "Long-RVOS" else -8, 36 if ds == "Long-RVOS" else 26)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("J&F difference to A0 top-1, with 95% video-level bootstrap interval")
for ax in axs[0]: ax.set_axisbelow(True)
fig.suptitle("Candidate-selection arms: J&F, and the difference to the top-1 start", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
foot(fig, "Dashed lines are oracle starts, a diagnostic ceiling only (not method performance). No arm's interval excludes zero on either dataset; no arm meets the Amendment 6 claim rule.\nA3 = offline multi-frame look-ahead over N = 5 or 10 frames. The best arm flips with the dataset (A3-5 on MeViS, A1 on Long-RVOS).\n" + NOTE, y=0.01)
save(fig, "fig2_selection_arms")

# ---------- Figure 3: initial vs later failure runs ----------
rows = [("First failure run of a sentence", None),
        ("Initial grounding", ("init",)), ("Mask propagation", ("prop",)), ("Re-identification", ("reid",)),
        ("Later failure runs (after a good start)", None),
        ("Re-identification ", ("reid",)), ("Identity drift", ("drift",)), ("Mask propagation ", ("prop",))]
cnt = {"MeViS": {"first": {"init": 127, "prop": 11, "reid": 0}, "later": {"reid": 11, "drift": 9, "prop": 32}},
       "Long-RVOS": {"first": {"init": 21, "prop": 2, "reid": 3}, "later": {"reid": 24, "drift": 0, "prop": 4}}}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True)
for ax, ds in zip(axs, ("MeViS", "Long-RVOS")):
    tot = sum(cnt[ds]["first"].values()) + sum(cnt[ds]["later"].values())
    c = DS[ds]; ypos = []; ylab = []; y = 0
    grp = "first"
    for lab, key in rows:
        if key is None:
            ax.text(0.5, y, lab, fontsize=9, fontweight="bold", color=INK, ha="left", va="center")
            grp = "first" if lab.startswith("First") else "later"; y += 1; continue
        v = cnt[ds][grp][key[0]]
        ax.barh(y, 100 * v / tot, height=0.55, color=c, zorder=3)
        ax.text(100 * v / tot + 1.0, y, f"{v}  ({100*v/tot:.0f}%)", va="center", fontsize=9, color=INK)
        ypos.append(y); ylab.append(lab.strip()); y += 1
    ax.set_yticks(ypos); ax.set_yticklabels(ylab); ax.set_ylim(7.6, -0.8)
    ax.set_xlim(0, 92); ax.set_xticks(range(0, 81, 20)); ax.set_xticklabels([f"{v}%" for v in range(0, 81, 20)])
    ax.grid(axis="y", visible=False)
    ax.set_title(f"{ds}: {tot} failure runs", loc="left", fontsize=11, fontweight="bold", color=INK, pad=10)
    ax.set_xlabel("Share of this dataset's failure runs")
fig.suptitle("MeViS fails at the start; Long-RVOS fails after a good start (mostly re-identification)", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.subplots_adjust(left=0.14, right=0.98, top=0.86, bottom=0.2, wspace=0.55)
foot(fig, "A0 (top-1 start, no memory), sentences visible at frame 0. A failure run = 3 or more consecutive visible frames with IoU < 0.5. Counts are runs, not sentences:\nMeViS 138 sentences have a first failure run; Long-RVOS 26. On MeViS, 62% of the later runs come from one video (329909730c21).\n" + NOTE, y=0.06)
save(fig, "fig3_initial_vs_later_failures")

# ---------- Figure 4: quality vs cost ----------
# overhead = (tracking time + offline selection time where measured) / V0 tracking time - 1
V0t = {"MeViS": 4430.2, "Long-RVOS": 6391.3}
sel = {"MeViS": {"A1": 68.7, "A3-5": 4046, "A3-10": 4046}, "Long-RVOS": {"A1": 20.6, "A3-5": 868, "A3-10": 868}}
trk = {"MeViS": {"V0": 4430.2, "A1": 4371.4, "A3-5": 4631.9, "A3-10": 4420.3, "Largest": 4462.3, "V1 memory": 5347.8, "V2 +TCS": 5415.3, "V4 re-ID": 5782.6},
       "Long-RVOS": {"V0": 6391.3, "A1": 6244.3, "A3-5": 7037.6, "A3-10": 6781.9, "Largest": 7406.0, "V1 memory": 8027.5, "V2 +TCS": 7926.9, "V4 re-ID": 8833.0}}
q = {"MeViS": {"V0": 26.88, "A1": 27.06, "A3-5": 32.16, "A3-10": 31.51, "Largest": 28.16, "V1 memory": 25.93, "V2 +TCS": 25.93, "V4 re-ID": 25.96},
     "Long-RVOS": {"V0": 30.38, "A1": 37.77, "A3-5": 35.12, "A3-10": 33.58, "Largest": 26.85, "V1 memory": 29.94, "V2 +TCS": 29.94, "V4 re-ID": 28.91}}
off = {"MeViS": {"A3-5": (-4, 9), "A3-10": (-8, -12), "A1": (12, 3), "V0": (-9, -12), "Largest": (10, 4), "V1 memory": (2, -14), "V2 +TCS": (0, 0), "V4 re-ID": (8, 4)},
       "Long-RVOS": {"A3-5": (8, 4), "A3-10": (8, -6), "A1": (10, 0), "V0": (-9, -12), "Largest": (8, -2), "V1 memory": (-6, 10), "V2 +TCS": (0, 0), "V4 re-ID": (-8, 9)}}
fig, axs = plt.subplots(1, 2, figsize=(11, 4.9))
for ax, ds in zip(axs, ("MeViS", "Long-RVOS")):
    c = DS[ds]
    ax.axvspan(-15, 15, color=BAND, zorder=1)
    for name in q[ds]:
        t = trk[ds][name] + sel[ds].get(name, 0)
        x = 100 * (t / V0t[ds] - 1)
        if ds == "Long-RVOS" and name == "V4 re-ID":
            x = 54.1   # V4 tracked 10 videos; overhead vs V0 on the same 10 videos, as in the cost table
        hollow = name == "Largest"
        ax.plot([x], [q[ds][name]], "o", ms=9, mfc=SURF if hollow else c, mec=c, mew=2 if hollow else 2.2, zorder=4)
        lab = {"V1 memory": "V1 memory / V2 +TCS", "V2 +TCS": None}.get(name, name)
        if lab:
            dx, dy = off[ds][name]
            ax.annotate(lab, (x, q[ds][name]), xytext=(dx, dy), textcoords="offset points", fontsize=9, color=INK2,
                        ha="right" if dx < 0 else "left", va="bottom" if dy > 0 else "top")
    ax.set_xlim(-12, 105 if ds == "MeViS" else 66)
    ax.set_ylim(24, 34 if ds == "MeViS" else 40)
    ax.set_title(ds, loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.set_xlabel("Extra time vs V0 (tracking + offline selection, %)")
    ax.set_xticks(range(0, 101, 20) if ds == "MeViS" else range(0, 61, 10))
    ax.set_xticklabels([f"{v}%" for v in ax.get_xticks()])
    ax.text(1.5, ax.get_ylim()[1] - 0.2, "within run-to-run spread", fontsize=8, color=MUTED, ha="center", va="top")
axs[0].set_ylabel("J&F")
fig.suptitle("Memory, TCS and re-ID add 21-54% time with no J&F gain; the A3 gain on MeViS needs an offline pass", x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(rect=(0, 0.17, 1, 0.94))
foot(fig, "Grey band: differences under about 15% are not interpreted (the same V0 tracker differed by about 5% on MeViS and 13% on Long-RVOS between runs).\nA1 and A3 include their offline selection time (A3: one SAM 2 look-ahead pass shared by N = 5 and N = 10, counted in full for each). Hollow marker: largest box, selection time not measured.\nV4 re-ID on Long-RVOS tracked 10 of 11 videos; its +54.1% is against V0 on the same 10 videos. V3 and V4+VLM are left out (one live video per dataset, no timing file).\nPeak memory: V0 about 0.7-0.8 GB; V1 about 0.8-0.9 GB; V2 about 1.4 GB; V4 re-ID about 1.7 GB. " + NOTE, y=0.06)
save(fig, "fig4_quality_vs_cost")
print("done")
