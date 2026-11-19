"""Closing the camera loop for closed-loop policies (after Stage 7).

Stage 3's `VisionPolicy` takes one snapshot at the start, which is enough for open-loop
policies. The learned policies re-read the cube's position every 50 ms, including while they
carry it, so they need the cube TRACKED from images at every policy step.

`CubeTracker` answers the same three questions the simulator used to answer, from images:
cube pose, cube yaw and target position. Per policy step, in order:

  1. wrist camera (on the hand, looks down between the fingers): used when it sees the WHOLE
     cube;
  2. front camera (fixed, sees the whole table): used when the wrist camera does not;
  3. neither: if the gripper is holding something (finger opening matches the cube width,
     read from the finger encoders), the cube moves with the hand at the offset last seen;
     otherwise the last estimate is kept.

Measured before choosing this order (6 episodes, reactive policy, 640x480): the front camera
is accurate while the cube is free (median 0.5 mm) but 10 mm off while the hand is over it,
because the hand hides the cube's top face; the wrist camera stays at 0.4-0.6 mm throughout.

"Sees the whole cube": the estimated cube's corners, projected back into that camera's image,
all lie inside it. Added after the first full run: when the cube starts near the edge of the
wrist camera's view, it sees part of the cube and the estimate is off by up to 20 mm (two
episodes failed on it; seeds 14 and 34 of the outside-area condition).

The target is located once, by the front camera, at the start of the episode (it does not
move, and a cube placed on it would later hide it).

Still from the simulator, as in Stage 3: robot joint angles and finger opening
(proprioception: a real robot has encoders for these). Images are rendered without shadows
or reflections, for speed (40 ms per 640x480 RGB-D frame on CPU).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from ari.perception.camera import RGBDCamera
from ari.perception.detect import localize_cube, localize_target
from ari.robot.transforms import make_T, rot_z

HOLD_TOLERANCE = 0.006       # m: finger opening within this of the cube width = holding


@dataclass
class TrackStats:
    """Per episode. Errors are against simulator truth and are for evaluation only."""
    steps: int = 0
    source: dict = field(default_factory=lambda: {"wrist": 0, "front": 0, "held": 0, "kept": 0})
    errors_mm: list = field(default_factory=list)
    target_error_mm: float | None = None
    target_found: bool = False

    def summary(self) -> dict:
        e = np.array(self.errors_mm) if self.errors_mm else np.array([np.nan])
        return {"steps": self.steps, "source": dict(self.source),
                "median_error_mm": float(np.nanmedian(e)), "p95_error_mm": float(np.nanpercentile(e, 95)),
                "max_error_mm": float(np.nanmax(e)), "target_error_mm": self.target_error_mm,
                "target_found": self.target_found}


def _plain(cam: RGBDCamera) -> RGBDCamera:
    for r in (cam.rgb_renderer, cam.depth_renderer):
        r.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        r.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = 0
    return cam


class CubeTracker:
    def __init__(self, env, width: int = 640, height: int = 480, use_wrist: bool = True, use_front: bool = True):
        self.env, self.use_wrist, self.use_front = env, use_wrist, use_front
        self.cams = {}
        if use_wrist:
            self.cams["wrist"] = _plain(RGBDCamera(env.model, "wrist_cam", width, height))
        self.cams["front"] = _plain(RGBDCamera(env.model, "front_cam", width, height))   # also finds the target
        self.reset()

    def reset(self):
        """Call after env.reset(): locates the target and the cube from the front camera."""
        self.stats = TrackStats()
        self._t = None
        self.position, self.yaw, self.offset, self.yaw_offset = None, 0.0, None, 0.0
        rgb, depth, cam, _ = self.cams["front"].capture(self.env.data)
        tgt = localize_target(rgb, depth, cam)
        true_tgt = self.env.data.site_xpos[self.env.target_site].copy()
        self.stats.target_found = bool(tgt.ok)
        if tgt.ok:
            self.target = np.array([tgt.position[0], tgt.position[1], true_tgt[2]])
            self.stats.target_error_mm = float(1000 * np.linalg.norm(self.target[:2] - true_tgt[:2]))
        else:
            self.target = None                # policies then see the true target; reported in stats
        self.update(force=True)

    def _holding(self) -> bool:
        return abs(self.env.finger_width() - 2 * self.env.cfg.cube_half_size) < HOLD_TOLERANCE

    def update(self, force: bool = False):
        """Re-estimate at most once per simulated instant (several queries per policy step)."""
        t = float(self.env.data.time)
        if not force and self._t == t:
            return
        self._t = t
        env = self.env
        T_tcp = env.tcp_pose()
        est, source = None, None
        for name in ("wrist", "front"):
            if name not in self.cams or (name == "front" and not self.use_front):
                continue
            rgb, depth, cam, _ = self.cams[name].capture(env.data)
            e = localize_cube(rgb, depth, cam, env.cfg.cube_half_size)
            if e.ok and fully_in_view(e.position, env.cfg.cube_half_size, cam):
                est, source = e, name
                break
        holding = self._holding()
        if est is not None:
            self.position, self.yaw = est.position.copy(), float(est.yaw)
            if holding:                                   # remember where it sits in the hand
                self.offset = self.position - T_tcp[:3, 3]
                self.yaw_offset = self.yaw - _yaw(T_tcp)
        elif holding and self.offset is not None:
            self.position = T_tcp[:3, 3] + self.offset
            self.yaw = _yaw(T_tcp) + self.yaw_offset
            source = "held"
        else:
            source = "kept"
        if self.position is None:                         # never seen at all: nothing better to offer
            self.position = np.array(env.data.xpos[env.cube_body]) + np.nan
        if not holding:
            self.offset = None
        self.stats.steps += 1
        self.stats.source[source] += 1
        truth = env.data.xpos[env.cube_body]
        self.stats.errors_mm.append(float(1000 * np.linalg.norm(self.position - truth)))

    # ---- what policies read ----
    def cube_pose(self) -> np.ndarray:
        self.update()
        return make_T(rot_z(self.yaw), self.position.copy())

    def cube_yaw(self) -> float:
        self.update()
        return float(self.yaw)

    def target_pos(self) -> np.ndarray | None:
        return None if self.target is None else self.target.copy()

    def close(self):
        for c in self.cams.values():
            c.close()


def fully_in_view(position: np.ndarray, half: float, cam, margin_px: float = 2.0) -> bool:
    """True if all 8 corners of a cube at `position` project inside the image (and in front of the camera)."""
    corners = np.array([[x, y, z] for x in (-half, half) for y in (-half, half) for z in (-half, half)]) + position
    uv, depth = cam.project(corners)
    return bool(np.all(depth > 0) and np.all(uv[:, 0] >= margin_px) and np.all(uv[:, 0] <= cam.width - margin_px)
                and np.all(uv[:, 1] >= margin_px) and np.all(uv[:, 1] <= cam.height - margin_px))


def _yaw(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))
