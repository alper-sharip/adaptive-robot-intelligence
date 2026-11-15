"""Stage 5: learning curves from the training logs in outputs/models/rl_*.json.

Three panels, one shared measure (training success rate of the exploring policy, last 100
episodes): reaching from scratch, pick-and-place from scratch, and residual RL on the clone.

Run:  python scripts/plot_rl.py      -> outputs/rl_learning_curves.png
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

M = Path("outputs/models")
BLUE, ORANGE = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated colour-blind safe)
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def curve(name):
    log = json.loads((M / f"rl_{name}.json").read_text())["log"]
    return [r["steps"] / 1000 for r in log], [100 * r["success"] for r in log]


def style(ax, title):
    ax.set_title(title, loc="left", fontsize=11, color=INK, pad=10)
    ax.set_ylim(-2, 102)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0, labelsize=9)
    ax.set_xlabel("environment steps (thousands)", color=MUTED, fontsize=9)


fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), facecolor=SURFACE)
for ax in axes:
    ax.set_facecolor(SURFACE)

ax = axes[0]
for seed in (0, 1, 2):
    if (M / f"rl_reach_s{seed}.json").exists():
        ax.plot(*curve(f"reach_s{seed}"), color=BLUE, lw=2, alpha=0.9)
style(ax, "Reach a point, from scratch (3 seeds)")
ax.set_ylabel("training success, last 100 episodes (%)", color=MUTED, fontsize=9)

ax = axes[1]
for name, color, label in (("pick_scratch", BLUE, "first reward"), ("pick_scratch_v2", ORANGE, "corrected reward")):
    if (M / f"rl_{name}.json").exists():
        ax.plot(*curve(name), color=color, lw=2, label=label)
ax.text(300, 8, "0% throughout, both runs", fontsize=9, color=INK, ha="center")
style(ax, "Pick-and-place, from scratch")
ax.legend(frameon=False, fontsize=8.5, loc="upper right", labelcolor=INK)

ax = axes[2]
series = [("residual100", BLUE, "trained on nominal episodes"), ("residual100_dist", ORANGE, "trained with 30% cube moved")]
for name, color, label in series:
    if (M / f"rl_{name}.json").exists():
        x, y = curve(name)
        ax.plot(x, y, color=color, lw=2, label=label)
ax.axvspan(0, 3 * 4.096, color=GRID, alpha=0.6, lw=0)
ax.text(3 * 4.096 + 4, 4, "critic warm-up:\npolicy frozen", fontsize=8, color=MUTED, va="bottom")
style(ax, "Residual RL on the 100-demo clone")
ax.legend(frameon=False, fontsize=8.5, loc="lower right", labelcolor=INK)

fig.tight_layout()
out = Path("outputs/rl_learning_curves.png")
fig.savefig(out, dpi=110, facecolor=SURFACE)
print(f"Saved {out}")
