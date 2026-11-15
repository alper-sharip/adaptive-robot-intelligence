"""M11: show what the perception pipeline sees, step by step, for one episode.

Panels: camera image with detections | depth | red mask (incl. the floor reflection) |
top-down view of the cube's 3-D points with estimated vs true cube outline.

Run:  python scripts/perception_demo.py --seed 3 --noise realistic
      MUJOCO_GL=egl python scripts/perception_demo.py ...   (headless Linux / Colab)
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ari.envs.pick_place import PickPlaceEnv, SceneConfig
from ari.perception.camera import RGBDCamera
from ari.perception.detect import RED, color_mask, fold_yaw, largest_component, localize_cube, localize_target
from ari.perception.noise import NOISE_LEVELS

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=3)
parser.add_argument("--noise", choices=list(NOISE_LEVELS), default="realistic")
parser.add_argument("--wall", action="store_true")
parser.add_argument("--out", type=Path, default=None)
args = parser.parse_args()

env = PickPlaceEnv(SceneConfig.with_wall() if args.wall else SceneConfig())
env.reset(args.seed)
cam = RGBDCamera(env.model)
rgb, depth, cm, _ = cam.capture(env.data)
rgb, depth = NOISE_LEVELS[args.noise].apply(rgb, depth, np.random.default_rng(args.seed + 100_000))
cube = localize_cube(rgb, depth, cm, env.cfg.cube_half_size)
target = localize_target(rgb, depth, cm)

true_c, true_yaw = env.cube_pose()[:3, 3], env.cube_yaw()
print(f"cube   : estimated {np.round(cube.position, 4)}  true {np.round(true_c, 4)}  "
      f"error {1000 * np.linalg.norm(cube.position - true_c):.2f} mm, "
      f"yaw error {np.degrees(abs(fold_yaw(cube.yaw - true_yaw))):.2f} deg")
print(f"target : estimated {np.round(target.position[:2], 4)}  true {np.round(env.target_pos()[:2], 4)}  "
      f"error {1000 * np.linalg.norm(target.position[:2] - env.target_pos()[:2]):.2f} mm")


def square(center, yaw, half):
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.array([[1, 1], [-1, 1], [-1, -1], [1, -1], [1, 1]]) * half
    return corners @ np.array([[c, s], [-s, c]]) + center[:2]


overlay = rgb.copy()
for (cen, yaw, z), col in (((cube.position, cube.yaw, 0.04), (255, 255, 0)),):
    sq = np.column_stack([square(cen, yaw, env.cfg.cube_half_size), np.full(5, z)])
    uv, _ = cm.project(sq)
    cv2.polylines(overlay, [uv.astype(np.int32)], True, col, 2)
uv, _ = cm.project(target.position)
cv2.drawMarker(overlay, tuple(int(x) for x in uv[0]), (255, 255, 0), cv2.MARKER_CROSS, 18, 2)

raw = color_mask(rgb, RED)
r, c = np.nonzero(largest_component(raw) & (depth > 0))
pts = cm.backproject_pixels(r, c, cv2.medianBlur(depth.astype(np.float32), 5))

fig, ax = plt.subplots(1, 4, figsize=(19, 4.2))
ax[0].imshow(overlay)
ax[0].set_title("camera + detections (yellow)")
ax[1].imshow(np.where(depth > 0, depth, np.nan), cmap="viridis")
ax[1].set_title(f"depth (noise: {args.noise})")
ax[2].imshow(raw, cmap="gray")
ax[2].set_title("red colour mask (raw)")
for a in ax[:3]:
    a.axis("off")
sc = ax[3].scatter(pts[:, 1] * 1000, pts[:, 0] * 1000, c=pts[:, 2] * 1000, s=4, cmap="plasma", vmin=0, vmax=45)
t = square(true_c, true_yaw, env.cfg.cube_half_size) * 1000
e = square(cube.position, cube.yaw, env.cfg.cube_half_size) * 1000
ax[3].plot(t[:, 1], t[:, 0], "k-", lw=2, label="true top face")
ax[3].plot(e[:, 1], e[:, 0], "c--", lw=2, label="estimated")
ax[3].set_xlabel("y (mm)")
ax[3].set_ylabel("x (mm)")
ax[3].set_aspect("equal")
ax[3].legend(fontsize=8, loc="upper right")
ax[3].set_title("cube points, top-down (colour = height, mm)")
fig.colorbar(sc, ax=ax[3], fraction=0.046)
fig.tight_layout()
out = args.out or Path(f"outputs/perception_seed{args.seed}_{args.noise}.png")
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=95)
print(f"Saved {out}")
