"""M12: feed perception into the existing policies.

`VisionPolicy` wraps any pick-and-place policy (scripted or planned). At the start of an
episode, with the arm at home, it takes ONE RGB-D snapshot, estimates the cube pose and the
target position, and hands the wrapped policy an `EnvView`: the real environment, except that
cube_pose(), cube_yaw() and target_pos() return the ESTIMATES instead of simulator truth.
The policies themselves are unchanged, so any difference in results is due to perception.

Still taken from the simulator (stated so nobody mistakes this for full autonomy):
  * the wall's geometry (for the planner's collision checks);
  * the cube's pose in the hand after grasping (CollisionChecker.attach measures it);
  * robot joint angles (proprioception; a real robot has encoders for these).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ari.perception.camera import RGBDCamera
from ari.perception.detect import CubeEstimate, TargetEstimate, localize_cube, localize_target
from ari.perception.noise import NOISE_LEVELS, NoiseModel
from ari.robot.transforms import make_T, rot_z


class EnvView:
    """Proxy for PickPlaceEnv that answers object queries from perception."""

    def __init__(self, env, cube: CubeEstimate, target: TargetEstimate):
        self._env, self._cube, self._target = env, cube, target

    def __getattr__(self, name):
        return getattr(self._env, name)

    def cube_pose(self) -> np.ndarray:
        return make_T(rot_z(self._cube.yaw), self._cube.position)

    def cube_yaw(self) -> float:
        return float(self._cube.yaw)

    def target_pos(self) -> np.ndarray:
        return self._target.position.copy()


@dataclass
class PerceptionErrors:
    cube_xy: float = np.nan     # m
    cube_z: float = np.nan      # m (signed)
    cube_yaw: float = np.nan    # rad, folded
    target_xy: float = np.nan   # m


class VisionPolicy:
    def __init__(self, base, noise: str | NoiseModel = "none", noise_seed: int = 0):
        self.base = base
        self.noise = NOISE_LEVELS[noise] if isinstance(noise, str) else noise
        self.noise_seed = noise_seed
        self._camera = None
        self._own_failure = ""

    # pass-throughs used by run_episode
    @property
    def phase(self):
        return self.base.phase if not self._own_failure else "done"

    @property
    def failure_reason(self):
        return self._own_failure or getattr(self.base, "failure_reason", "")

    @property
    def plan_time(self):
        return getattr(self.base, "plan_time", 0.0)

    def reset(self, env) -> None:
        from ari.perception.detect import fold_yaw

        if self._camera is None or self._camera.model is not env.model:
            self._camera = RGBDCamera(env.model)
        rgb, depth, cam, _ = self._camera.capture(env.data)
        rgb, depth = self.noise.apply(rgb, depth, np.random.default_rng(self.noise_seed))
        self.cube_est = localize_cube(rgb, depth, cam, env.cfg.cube_half_size)
        self.target_est = localize_target(rgb, depth, cam)
        self.last_images = (rgb, depth)
        self._own_failure = ""
        if not self.cube_est.ok:
            self._own_failure = f"cube_{self.cube_est.reason}"
            return
        if not self.target_est.ok:
            self._own_failure = f"target_{self.target_est.reason}"
            return
        true_cube = env.cube_pose()[:3, 3]
        self.errors = PerceptionErrors(
            cube_xy=float(np.linalg.norm(self.cube_est.position[:2] - true_cube[:2])),
            cube_z=float(self.cube_est.position[2] - true_cube[2]),
            cube_yaw=float(abs(fold_yaw(self.cube_est.yaw - env.cube_yaw()))),
            target_xy=float(np.linalg.norm(self.target_est.position[:2] - env.target_pos()[:2])),
        )  # for evaluation only; the policy never sees these
        self.view = EnvView(env, self.cube_est, self.target_est)
        self.base.reset(self.view)

    def act(self, env, dt, q_cmd=None):
        if self._own_failure:
            return None
        return self.base.act(self.view, dt, q_cmd=q_cmd)
