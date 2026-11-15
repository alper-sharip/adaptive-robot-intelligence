"""Stage 6: a scene with several cubes and several target zones.

Three cubes (red, blue, yellow) and three target zones (red, blue, green). Cube and target
colours overlap deliberately: "the blue cube on the red target" and "the red cube on the blue
target" use the same words, so an instruction cannot be understood from its colour words alone.

Reuse through FOCUS. Every skill built in Stages 2-5 asks the environment the same questions:
where is the cube, where is the target, is the task done? `focus(cube, target)` chooses which
cube and which target those questions refer to. Language decides the focus; the skills
themselves run unchanged.

Cubes are sampled in the same table region as in Stage 4 and targets in the same target
region, so the learned skills stay inside the conditions they were trained on.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ari.envs.pick_place import PickPlaceEnv, SceneConfig
from ari.robot.ik import Q_HOME
from ari.robot.model import PANDA_SCENE, arm_qpos_indices

CUBE_COLORS = ("red", "blue", "yellow")
TARGET_COLORS = ("red", "blue", "green")
RGBA = {"red": (0.85, 0.2, 0.2), "blue": (0.2, 0.35, 0.9), "yellow": (0.95, 0.8, 0.1), "green": (0.2, 0.8, 0.3)}
MIN_CUBE_SEPARATION = 0.10      # fingers (open 8 cm) must fit beside a neighbouring cube
MIN_TARGET_SEPARATION = 0.09


@dataclass
class MultiEpisode:
    seed: int
    cube_xy: np.ndarray      # (3, 2)
    cube_yaw: np.ndarray     # (3,)
    target_xy: np.ndarray    # (3, 2)


def _sample_points(rng, region, n, min_sep):
    (x0, x1), (y0, y1) = region
    while True:
        pts = rng.uniform([x0, y0], [x1, y1], size=(n, 2))
        d = np.linalg.norm(pts[:, None] - pts[None], axis=-1) + np.eye(n)
        if d.min() >= min_sep:
            return pts


class MultiObjectEnv(PickPlaceEnv):
    def __init__(self, cfg: SceneConfig | None = None):
        from ari.perception.camera import look_at_quat

        self.cfg = cfg = cfg or SceneConfig()
        spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
        h = cfg.cube_half_size
        for i, color in enumerate(CUBE_COLORS):
            body = spec.worldbody.add_body(name=f"cube_{color}", pos=[0.5, -0.2 + 0.15 * i, h])
            body.add_freejoint(name=f"cube_{color}_joint")
            body.add_geom(name=f"cube_{color}_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[h] * 3,
                          mass=cfg.cube_mass, friction=[cfg.cube_friction, 0.005, 0.0001],
                          rgba=[*RGBA[color], 1.0], priority=1)
        for j, color in enumerate(TARGET_COLORS):
            spec.worldbody.add_site(name=f"target_{color}", pos=[0.5, 0.1 + 0.1 * j, 0.0005],
                                    type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[cfg.target_radius, 0.0005, 0],
                                    rgba=[*RGBA[color], 0.55])
        spec.worldbody.add_camera(name="front_cam", pos=list(cfg.camera_pos),
                                  quat=list(look_at_quat(cfg.camera_pos, cfg.camera_target)), fovy=cfg.camera_fovy)
        self.model = spec.compile()
        if cfg.elliptic_cone:
            self.model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        self.model.opt.impratio = cfg.impratio
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm_qpos = arm_qpos_indices(m)
        self.hand_body = m.body("hand").id
        self.wall_geom = None
        self.cube_bodies = [m.body(f"cube_{c}").id for c in CUBE_COLORS]
        self.cube_qposadr = [m.joint(f"cube_{c}_joint").qposadr[0] for c in CUBE_COLORS]
        self.cube_dofadrs = [m.joint(f"cube_{c}_joint").dofadr[0] for c in CUBE_COLORS]
        self.target_sites = [m.site(f"target_{c}").id for c in TARGET_COLORS]
        self.episode = None
        self.focus(0, 0)

    # ---------- focus: which cube / target the single-object interface refers to ----------
    def focus(self, cube, target) -> None:
        i = CUBE_COLORS.index(cube) if isinstance(cube, str) else int(cube)
        j = TARGET_COLORS.index(target) if isinstance(target, str) else int(target)
        self.cube_index, self.target_index = i, j
        self.cube_body, self.cube_qpos, self.cube_dofadr = self.cube_bodies[i], self.cube_qposadr[i], self.cube_dofadrs[i]
        self.target_site = self.target_sites[j]

    # ---------- episodes ----------
    def sample_episode(self, seed: int) -> MultiEpisode:
        rng = np.random.default_rng(seed)
        while True:
            cubes = _sample_points(rng, self.cfg.cube_region, len(CUBE_COLORS), MIN_CUBE_SEPARATION)
            targets = _sample_points(rng, self.cfg.target_region, len(TARGET_COLORS), MIN_TARGET_SEPARATION)
            # Every cube at least cfg.min_separation (12 cm) from every target, as in Stage 4.
            # Without this the learned skill failed 3/45 episodes, all with a cube 9-10 cm from
            # its target: closer than anything in its training data (devlog.md).
            if np.linalg.norm(cubes[:, None] - targets[None], axis=-1).min() >= self.cfg.min_separation:
                break
        return MultiEpisode(seed, cubes, rng.uniform(-np.pi, np.pi, len(CUBE_COLORS)), targets)

    def reset(self, seed: int) -> MultiEpisode:
        ep = self.sample_episode(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qpos] = Q_HOME
        d.qpos[[self.arm_qpos[-1] + 1, self.arm_qpos[-1] + 2]] = 0.04
        for i, adr in enumerate(self.cube_qposadr):
            d.qpos[adr:adr + 3] = [*ep.cube_xy[i], self.cfg.cube_half_size]
            d.qpos[adr + 3:adr + 7] = [np.cos(ep.cube_yaw[i] / 2), 0, 0, np.sin(ep.cube_yaw[i] / 2)]
        for j, site in enumerate(self.target_sites):
            m.site_pos[site, :2] = ep.target_xy[j]
        d.ctrl[:7] = Q_HOME
        d.ctrl[7] = 255
        mujoco.mj_forward(m, d)
        self.episode = ep
        return ep

    # ---------- queries about any object (the planner's world model) ----------
    def cube_position(self, i: int) -> np.ndarray:
        return self.data.xpos[self.cube_bodies[i]].copy()

    def cube_yaw_of(self, i: int) -> float:
        R = self.data.xmat[self.cube_bodies[i]].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def target_position(self, j: int) -> np.ndarray:
        return self.data.site_xpos[self.target_sites[j]].copy()

    def is_placed(self, cube, target) -> bool:
        """Is this cube resting on this target? (The post-condition of a 'put X on Y' step.)"""
        i = CUBE_COLORS.index(cube) if isinstance(cube, str) else int(cube)
        j = TARGET_COLORS.index(target) if isinstance(target, str) else int(target)
        c, t = self.cube_position(i), self.target_position(j)
        return bool(c[2] < self.cfg.cube_half_size + 0.005 and np.linalg.norm(c[:2] - t[:2]) < self.cfg.target_radius)

    def cube_on_target(self, target) -> int | None:
        """Index of the cube currently resting on this target, or None."""
        for i in range(len(CUBE_COLORS)):
            if self.is_placed(i, target):
                return i
        return None

    def teleport_cube(self, i: int, xy) -> None:
        adr, dof = self.cube_qposadr[i], self.cube_dofadrs[i]
        self.data.qpos[adr:adr + 2] = xy
        self.data.qpos[adr + 2] = self.cfg.cube_half_size
        self.data.qvel[dof:dof + 6] = 0
        mujoco.mj_forward(self.model, self.data)
