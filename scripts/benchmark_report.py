"""M22: turn outputs/benchmark/results.json into the figures and tables of docs/results_stage7.md.

    python scripts/benchmark_report.py          # writes outputs/benchmark_heatmap.png, outputs/benchmark_sweeps.png,
                                                # outputs/benchmark/summary.md (tables, ready to paste)
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from ari import benchmark as B

# Colours: one blue ramp for magnitude (light = low); six fixed categorical colours for the
# policies in the line charts, checked for colour-vision-deficiency separation in this order.
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#eef4fd", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
MARKERS = ["o", "s", "^", "D", "v", "P"]

# Display order and grouping of the conditions.
GROUPS = [("Training", ["nominal"]),
          ("Disturbance", ["moved", "latency"]),
          ("Sensing", ["jitter", "offset"]),
          ("Scene", ["outside", "small", "large", "heavy"]),
          ("Grip physics", ["friction07", "friction05", "stiff", "friction05_stiff"])]
ORDER = [c for _, cs in GROUPS for c in cs]


def load():
    return json.loads(Path("outputs/benchmark/results.json").read_text())


def rate(store, spec, cond):
    r = store.get(f"{spec}|{cond}")
    return None if r is None else 100 * r["success_rate"]


def heatmap(store, path):
    rows = list(B.POLICIES.items())
    labels = [v[0] for _, v in rows]
    specs = [k for k, _ in rows]
    grid = np.array([[np.nan if rate(store, s, c) is None else rate(store, s, c) for c in ORDER] for s in specs])
    mean = np.array([np.nanmean(g) if not np.isnan(g).any() else np.nan for g in grid])

    cmap = LinearSegmentedColormap.from_list("blue", RAMP)
    fig, ax = plt.subplots(figsize=(13.2, 0.62 * len(specs) + 2.3), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    x_of, x = {}, 0.0
    for _, cs in GROUPS:                       # a wider gap between groups
        for c in cs:
            x_of[c] = x
            x += 1.0
        x += 0.35
    x_mean = x + 0.1
    for i in range(len(specs)):
        for j, c in enumerate(ORDER):
            v = grid[i, j]
            if np.isnan(v):
                ax.add_patch(plt.Rectangle((x_of[c] + 0.03, i + 0.04), 0.94, 0.92, facecolor=SURFACE, edgecolor=GRID, lw=1))
                ax.text(x_of[c] + 0.5, i + 0.5, "not run", ha="center", va="center", fontsize=7.5, color=MUTED)
                continue
            ax.add_patch(plt.Rectangle((x_of[c] + 0.03, i + 0.04), 0.94, 0.92, facecolor=cmap(v / 100), edgecolor="none"))
            ax.text(x_of[c] + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center", fontsize=11,
                    color="white" if v >= 55 else INK, fontweight="bold" if v >= 95 else "normal")
        if not np.isnan(mean[i]):
            ax.text(x_mean + 0.5, i + 0.5, f"{mean[i]:.0f}", ha="center", va="center", fontsize=11, color=INK, fontweight="bold")
    for c in ORDER:
        ax.text(x_of[c] + 0.5, -0.12, B.CONDITIONS[c].label.replace(", ", ",\n").replace(" ", "\n", 1) if len(B.CONDITIONS[c].label) > 9
                else B.CONDITIONS[c].label, ha="center", va="bottom", fontsize=8.5, color=INK, linespacing=1.15)
    ax.text(x_mean + 0.5, -0.12, "Mean of\n13", ha="center", va="bottom", fontsize=8.5, color=INK, linespacing=1.15)
    for name, cs in GROUPS:
        x0, x1 = x_of[cs[0]], x_of[cs[-1]] + 1
        ax.plot([x0 + 0.05, x1 - 0.05], [-1.05, -1.05], color=MUTED, lw=1)
        ax.text((x0 + x1) / 2, -1.13, name, ha="center", va="bottom", fontsize=9, color=MUTED)
    for i, lab in enumerate(labels):
        ax.text(-0.15, i + 0.5, lab, ha="right", va="center", fontsize=10, color=INK)
    ax.set_xlim(-3.4, x_mean + 1.1)
    ax.set_ylim(len(specs) + 0.05, -1.75)
    ax.axis("off")
    fig.suptitle("Success rate (%) of every policy under conditions it was not trained on", x=0.012, y=0.985, ha="left",
                 fontsize=13, color=INK, fontweight="bold")
    fig.text(0.012, 0.915, "100 seeded episodes per cell, same seeds for all. One change from the training conditions per column "
             "(two in the last).", ha="left", fontsize=9.5, color=MUTED)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.90, bottom=0.02)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return specs, labels, grid, mean


def sweeps(store, path):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.3), dpi=160, sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (axis, (attr, xlabel, values)) in zip(axes, B.SWEEPS.items()):
        ax.set_facecolor(SURFACE)
        base = getattr(B.CONDITIONS[axis], attr)
        xs = [0.0] + [50.0 * v if axis == "latency" else 1000.0 * v for v in values]
        for k, (spec, (label, _)) in enumerate(B.POLICIES.items()):
            ys = [rate(store, spec, "nominal")] + [rate(store, spec, axis if v == base else f"{axis}@{v}") for v in values]
            if any(y is None for y in ys):
                continue
            ax.plot(xs, ys, color=SERIES[k], lw=2, marker=MARKERS[k], ms=7, mec=SURFACE, mew=1.2, label=label, clip_on=False,
                    zorder=3 + k)
        if axis == "latency":   # the drop past 200 ms is the 20 s episode limit, not failure (results_stage7.md)
            ax.text(0.03, 0.25, "Past 200 ms the arm is slower and\nruns out of the 20 s episode limit.\nGiven 90 s: 30/30 at 300 and 500 ms.",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=MUTED, linespacing=1.3)
        ax.set_xlabel(xlabel, fontsize=10, color=INK)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{v:g}" for v in xs], fontsize=9, color=MUTED)
        ax.set_ylim(0, 103)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="y", labelsize=9, labelcolor=MUTED, length=0)
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("Success rate (%)", fontsize=10, color=INK)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=6, frameon=False, fontsize=9.5,
               handlelength=2.2, columnspacing=1.6)
    fig.suptitle("How far each perturbation can be pushed before a policy breaks", x=0.012, y=0.985, ha="left", fontsize=13,
                 color=INK, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.99, top=0.80, bottom=0.13, wspace=0.08)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def tables(store, specs, labels, grid, mean) -> str:
    out = ["## Main matrix: success rate (%), 100 episodes per cell\n",
           "| Policy | " + " | ".join(B.CONDITIONS[c].label for c in ORDER) + " | Mean | Worst |", "|---|" + "---|" * (len(ORDER) + 2)]
    for lab, g, m in zip(labels, grid, mean):
        cells = ["n/r" if np.isnan(v) else f"{v:.0f}" for v in g]
        out.append(f"| {lab} | " + " | ".join(cells) + f" | {'n/a' if np.isnan(m) else f'{m:.1f}'} | "
                   f"{'n/a' if np.isnan(g).any() else f'{np.nanmin(g):.0f}'} |")
    out += ["", "## Conditions\n", "| Condition | What differs from training |", "|---|---|"]
    out += [f"| {B.CONDITIONS[c].label} | {B.CONDITIONS[c].what} |" for c in ORDER]

    out += ["", "## Severity sweeps: success rate (%)\n"]
    for axis, (attr, xlabel, values) in B.SWEEPS.items():
        base = getattr(B.CONDITIONS[axis], attr)
        heads = ["0"] + [f"{(50 * v if axis == 'latency' else 1000 * v):g}" for v in values]
        out += [f"| {xlabel} | " + " | ".join(heads) + " |", "|---|" + "---|" * len(heads)]
        for spec, (label, _) in B.POLICIES.items():
            ys = [rate(store, spec, "nominal")] + [rate(store, spec, axis if v == base else f"{axis}@{v}") for v in values]
            out.append(f"| {label} | " + " | ".join("n/r" if y is None else f"{y:.0f}" for y in ys) + " |")
        out.append("")

    out += ["## Placement precision in the training conditions (successful episodes)\n",
            "| Policy | Median error (mm) | 95th percentile (mm) | Within 10 mm (% of all episodes) | Within 5 mm (% of all episodes) |",
            "|---|---|---|---|---|"]
    for spec, lab in zip(specs, labels):
        r = store.get(f"{spec}|nominal")
        if r is None:
            continue
        e = np.array([ep["placement_mm"] for ep in r["episodes"] if ep["success"]])
        if len(e) == 0:
            continue
        out.append(f"| {lab} | {np.median(e):.1f} | {np.percentile(e, 95):.1f} | {100 * np.sum(e < 10) / r['n']:.0f} | "
                   f"{100 * np.sum(e < 5) / r['n']:.0f} |")

    out += ["", "## Failure types, summed over the 13 conditions\n", "| Policy | Episodes failed | never lifted | "
            "still holding / off table | dropped or misplaced | placed, did not retreat |", "|---|---|---|---|---|---|"]
    kinds = ["never_lifted", "still_holding_or_off_table", "dropped_or_misplaced", "placed_but_did_not_retreat"]
    for spec, lab in zip(specs, labels):
        tot = {k: 0 for k in kinds}
        n = 0
        for c in ORDER:
            r = store.get(f"{spec}|{c}")
            if r is None:
                continue
            n += r["n"]
            for k, v in r["failures"].items():
                tot[k] += v
        out.append(f"| {lab} | {sum(tot.values())} of {n} | " + " | ".join(str(tot[k]) for k in kinds) + " |")

    img = [(c, store.get(f"{B.IMAGE_POLICY[0]}|{c}")) for c in B.IMAGE_CONDITIONS]
    if any(r for _, r in img):
        out += ["", "## Image policy (two 64x64 cameras), success rate (%)\n",
                "| " + " | ".join(B.CONDITIONS[c].label for c, _ in img) + " |", "|" + "---|" * len(img),
                "| " + " | ".join("n/r" if r is None else f"{100 * r['success_rate']:.0f} ({100 * r['ci95'][0]:.0f}-{100 * r['ci95'][1]:.0f})"
                                  for _, r in img) + " |"]

    if any("latency_legacy@" in k for k in store):
        out += ["", "## Latency before and after the fix: success rate (%)\n",
                "Old: steps relative to the pose when the action is applied. Fixed: relative to the pose the policy observed.\n",
                "| Policy | No delay | 50 ms old | 50 ms fixed | 100 ms old | 100 ms fixed | 150 ms old | 150 ms fixed | "
                "200 ms old | 200 ms fixed |", "|---|" + "---|" * 9]
        for spec, (label, _) in B.POLICIES.items():
            cells = [rate(store, spec, "nominal")]
            for k in (1, 2, 3, 4):
                cells += [rate(store, spec, f"latency_legacy@{k}"), rate(store, spec, "latency" if k == 2 else f"latency@{k}")]
            out.append(f"| {label} | " + " | ".join("n/r" if v is None else f"{v:.0f}" for v in cells) + " |")

    extras = [k for k in store if k.split("|")[0] == "reactive#4cm" or "heavy@" in k]
    if extras:
        out += ["", "## Footnote measurements\n", "| Cell | Success |", "|---|---|"]
        out += [f"| {k} | {store[k]['successes']}/{store[k]['n']} |" for k in extras]
    return "\n".join(out) + "\n"


def camera_tables(store, cam) -> str:
    """Camera loop vs simulator state, on the SAME seeds (the camera cells run seeds 0-49)."""
    out = ["## Camera loop: success with cube and target from images vs from the simulator\n",
           "Same policies, same seeds. Each cell: simulator state → cameras (episodes of 50); "
           "change in brackets.\n",
           "| Policy | " + " | ".join(B.CONDITIONS[c].label for c in B.CAMERA_CONDITIONS) + " |",
           "|---|" + "---|" * len(B.CAMERA_CONDITIONS)]
    for spec, (label, _) in B.POLICIES.items():
        cells = []
        for c in B.CAMERA_CONDITIONS:
            rc, rs = cam.get(f"{spec}|cam:{c}"), store.get(f"{spec}|{c}")
            if rc is None or rs is None:
                cells.append("n/r")
                continue
            seeds = {e["seed"] for e in rc["episodes"]}
            ks = sum(e["success"] for e in rs["episodes"] if e["seed"] in seeds)
            kc = rc["successes"]
            cells.append(f"{ks} → {kc} ({kc - ks:+d})")
        out.append(f"| {label} | " + " | ".join(cells) + " |")
    out += ["", "## Tracking accuracy (all camera episodes)\n",
            "| Condition | Median error (mm) | 95th percentile of per-episode 95th percentile (mm) | Wrist camera | Front camera | "
            "Moved with hand | Kept last | Target error, median (mm) |", "|---|---|---|---|---|---|---|---|"]
    for c in B.CAMERA_CONDITIONS:
        tr = [e["tracking"] for spec in B.POLICIES if (r := cam.get(f"{spec}|cam:{c}")) for e in r["episodes"]]
        if not tr:
            continue
        src = {k: sum(t["source"][k] for t in tr) for k in ("wrist", "front", "held", "kept")}
        tot = sum(src.values())
        out.append(f"| {B.CONDITIONS[c].label} | {np.median([t['median_error_mm'] for t in tr]):.2f} | "
                   f"{np.percentile([t['p95_error_mm'] for t in tr], 95):.2f} | "
                   + " | ".join(f"{100 * src[k] / tot:.1f}%" for k in ("wrist", "front", "held", "kept"))
                   + f" | {np.median([t['target_error_mm'] for t in tr if t['target_error_mm'] is not None]):.2f} |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    store = load()
    specs, labels, grid, mean = heatmap(store, Path("outputs/benchmark_heatmap.png"))
    sweeps(store, Path("outputs/benchmark_sweeps.png"))
    text = tables(store, specs, labels, grid, mean)
    cam_path = Path("outputs/benchmark/camera.json")
    if cam_path.exists():
        text += "\n" + camera_tables(store, json.loads(cam_path.read_text()))
    Path("outputs/benchmark/summary.md").write_text(text)
    print(text)
