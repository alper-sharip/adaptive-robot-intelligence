"""Scripted pick-and-place for the SO-101, on the closed-form top-down IK.

Waypoints for the tool point (above the cube, at the cube, lifted, above the target, placed,
retreat) are turned into joint angles by `ik_top_down`, and the servos follow minimum-jerk
joint trajectories between them. This is the SO-101 counterpart of Stage 2's scripted
baseline, and the reference to compare the real arm against.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ari.so101.env import SO101Env
from ari.so101.kinematics import ik_top_down

GRIPPER_OPEN = 0.55        # rad: about 60 mm between the jaw tips
GRIPPER_CLOSED = -0.17     # rad: the joint's lower limit, so the servo squeezes at its torque limit


@dataclass
class PickPlaceResult:
    seed: int
    success: bool
    failure: str = ""
    placement_error_mm: float = float("nan")
    duration_s: float = 0.0
    log: list = field(default_factory=list)
    held_offset: np.ndarray | None = None   # only with measure_held=True


def _min_jerk(s):
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _tool_axes(yaw: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontal axes of a top-down tool frame: y = (cos yaw, sin yaw, 0), z = x * y with x = down."""
    y = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.cross([0.0, 0.0, -1.0], y)
    return y, z


@dataclass
class GripperCalibration:
    """Offsets in the top-down tool frame's horizontal axes (y, z), in metres.
    held:   where a lifted cube sits relative to the tool point. The SO-101 has one fixed and one
            moving jaw, so closing pushes the cube toward the fixed jaw (measured: 9.7 mm).
    placed: where the cube ends up relative to the tool point after it is set down and released.
            Smaller than `held` (5.4 mm): when the moving jaw opens, the squeezed cube springs back.
    Both come from one uncompensated calibration episode, like a TCP calibration on hardware."""
    held: np.ndarray
    placed: np.ndarray


class ScriptedPickPlace:
    """Top-down scripted pick-and-place with closed-form IK and min-jerk joint moves.

    With a calibration, the tool is aimed at where the jaws will hold the cube, minus `clearance`
    (aiming at the full offset brings the fixed jaw down onto the cube instead of beside
    it: 0/40 lifted; 4 mm was the best of 0-6 mm), and the place point is shifted by the
    measured `placed` offset, so the cube ends up on the target."""

    def __init__(self, env: SO101Env, approach_h: float = 0.05, grasp_dz: float = 0.0, segment_s: float = 1.0,
                 calibration: GripperCalibration | None = None, clearance: float = 0.004):
        self.env, self.approach_h, self.grasp_dz, self.T = env, approach_h, grasp_dz, segment_s
        self.cal = calibration
        self.clearance = clearance
        self.last_yaw = 0.0

    def _offsets(self, yaw: float) -> tuple[np.ndarray, np.ndarray]:
        """World-frame shifts (grasp aim, place) for this grasp yaw."""
        if self.cal is None:
            return np.zeros(3), np.zeros(3)
        y, z = _tool_axes(yaw)
        held = np.asarray(self.cal.held, dtype=float)
        n = np.linalg.norm(held)
        aim = held * max(0.0, 1 - self.clearance / n) if n > 0 else held
        placed = np.asarray(self.cal.placed, dtype=float)
        return aim[0] * y + aim[1] * z, placed[0] * y + placed[1] * z

    @staticmethod
    def calibrate(env: SO101Env, seed: int = 999_999) -> GripperCalibration:
        """Run one uncompensated pick-and-place and measure both offsets."""
        pp = ScriptedPickPlace(env)
        res = pp.run(seed, measure_held=True)
        if res.held_offset is None:
            raise RuntimeError(f"calibration grasp failed: {res.failure}")
        y, z = _tool_axes(pp.last_yaw)
        d = env.cube_pos() - env.target_pos()
        return GripperCalibration(held=res.held_offset, placed=np.array([d @ y, d @ z]))

    def _yaw_for_cube(self, p: np.ndarray, cube_yaw: float) -> float:
        """The cube looks the same every 90 deg: pick the equivalent yaw the wrist can reach
        with the least roll, at the grasp point."""
        env = self.env
        best = None
        for k in range(4):
            yaw = (cube_yaw + k * np.pi / 2 + np.pi) % (2 * np.pi) - np.pi
            sol = ik_top_down(env.chain, env.geometry, p, yaw)
            if sol.ok and (best is None or abs(sol.q[4]) < abs(best[1].q[4])):
                best = (yaw, sol)
        if best is None:
            raise ValueError("no reachable grasp yaw")
        return best[0]

    def _move(self, q_goal, gripper, duration=None, frame_callback=None):
        env = self.env
        q0 = env.data.ctrl[env.arm_act].copy()
        n = max(1, int((duration or self.T) / env.model.opt.timestep))
        for i in range(1, n + 1):
            env.data.ctrl[env.arm_act] = q0 + _min_jerk(i / n) * (q_goal - q0)
            env.data.ctrl[env.gripper_act] = gripper
            env.step()
            if frame_callback is not None:
                frame_callback(env)

    def _hold(self, gripper, duration, frame_callback=None):
        self._move(self.env.data.ctrl[self.env.arm_act].copy(), gripper, duration, frame_callback)

    def run(self, seed: int, frame_callback=None, measure_held: bool = False) -> PickPlaceResult:
        env = self.env
        ep = env.reset(seed)
        h = env.cfg.cube_half_size
        cube = np.array([*ep.cube_xy, h + self.grasp_dz])
        place = np.array([*ep.target_xy, h + self.grasp_dz + 0.002])
        try:
            yaw = self._yaw_for_cube(cube, ep.cube_yaw)
        except ValueError as err:
            return PickPlaceResult(seed, False, f"unreachable: {err}")
        self.last_yaw = yaw
        aim, placed = self._offsets(yaw)
        cube = cube - aim          # aim so the cube is (almost) where the jaws will hold it
        place = place - placed     # so that the released cube settles on the target
        up = np.array([0, 0, self.approach_h])
        plan = []
        for name, p, grip, dur in (("above cube", cube + up, GRIPPER_OPEN, 1.5), ("at cube", cube, GRIPPER_OPEN, 0.8),
                                   ("grasp", None, GRIPPER_CLOSED, 0.5), ("lift", cube + up, GRIPPER_CLOSED, 0.8),
                                   ("above target", place + up, GRIPPER_CLOSED, 1.2), ("place", place, GRIPPER_CLOSED, 0.8),
                                   ("release", None, GRIPPER_OPEN, 0.4), ("retreat", place + up, GRIPPER_OPEN, 0.8)):
            if p is None:
                plan.append((name, None, grip, dur))
                continue
            sol = ik_top_down(env.chain, env.geometry, p, yaw)
            if not sol.ok:
                return PickPlaceResult(seed, False, f"unreachable at '{name}': {sol.reason}")
            plan.append((name, sol.q, grip, dur))
        log = []
        held_offset = None
        for name, q, grip, dur in plan:
            if q is None:
                self._hold(grip, dur, frame_callback)
            else:
                self._move(q, grip, dur, frame_callback)
            log.append((name, np.round(env.tool_pos(), 4).tolist(), np.round(env.cube_pos(), 4).tolist()))
            if name == "lift" and env.cube_pos()[2] < h + 0.02:
                return PickPlaceResult(seed, False, "not lifted", 1000 * env.placement_error(), env.data.time, log)
            if name == "lift" and measure_held:
                y, z = _tool_axes(yaw)
                d = env.cube_pos() - env.tool_pos()
                held_offset = np.array([d @ y, d @ z])
        self._move(env.q_rest, GRIPPER_OPEN, 1.0, frame_callback)
        self._hold(GRIPPER_OPEN, 0.5, frame_callback)
        ok = env.is_success()
        return PickPlaceResult(seed, ok, "" if ok else "misplaced", 1000 * env.placement_error(), env.data.time, log,
                               held_offset)
