"""M5/M9: the pick-and-place scene, with seeded randomization.

Scene: the Panda is mounted on a tabletop (the floor plane plays the table, so
table height = robot base height = 0). There is one cube and one target zone.

Everything random comes from `reset(seed)`, so episode N is the same episode
every time you run it, on any machine. Fixed seeds are what make "policy A
beats policy B" a checkable claim rather than a lucky draw.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ari.robot.ik import Q_HOME
from ari.robot.model import PANDA_SCENE, TCP_OFFSET, arm_qpos_indices
from ari.robot.transforms import make_T


@dataclass
class SceneConfig:
    cube_half_size: float = 0.02          # 4 cm cube
    cube_mass: float = 0.1                # kg
    cube_friction: float = 1.0            # sliding friction coefficient
    # Sampling regions (x range, y range) in metres, robot base frame.
    cube_region: tuple = ((0.40, 0.60), (-0.25, 0.05))
    target_region: tuple = ((0.40, 0.60), (0.10, 0.30))
    min_separation: float = 0.12          # cube and target at least this far apart
    target_radius: float = 0.03           # success if cube centre lands within this
    # Gripper servo stiffness multiplier. Menagerie's gripper is a position servo with
    # kp = 100 N/m on the finger tendon; closed on a 4 cm cube that squeezes with only
    # ~2 N, and the cube slowly slips (see docs/devlog.md). 1.0 = unmodified Menagerie.
    gripper_stiffness_scale: float = 1.0
    # Friction model. MuJoCo's default (pyramidal cone, impratio 1) lets a held object
    # creep ~1.5 mm/s under its own weight; the elliptic cone with impratio 10 is
    # MuJoCo's recommended setting for grasping and cuts the creep ~10x (devlog.md).
    elliptic_cone: bool = True
    impratio: float = 10.0


@dataclass
class EpisodeSpec:
    """Everything that defines one randomized episode."""
    seed: int
    cube_xy: np.ndarray
    cube_yaw: float
    target_xy: np.ndarray


def build_model(cfg: SceneConfig) -> mujoco.MjModel:
    """Menagerie Panda scene + cube + target marker, assembled with MjSpec."""
    spec = mujoco.MjSpec.from_file(str(PANDA_SCENE))
    cube = spec.worldbody.add_body(name="cube", pos=[0.5, 0.0, cfg.cube_half_size])
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(
        name="cube_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=[cfg.cube_half_size] * 3,
        mass=cfg.cube_mass,
        friction=[cfg.cube_friction, 0.005, 0.0001],
        rgba=[0.85, 0.2, 0.2, 1.0],
        # MuJoCo combines contact friction with max() unless priorities differ; priority 1
        # makes the CUBE's friction govern its contacts, so cube_friction actually matters.
        priority=1,
    )
    # Target: a flat, non-colliding disc (sites never collide).
    spec.worldbody.add_site(
        name="target",
        pos=[0.5, 0.2, 0.0005],
        type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        size=[cfg.target_radius, 0.0005, 0],
        rgba=[0.2, 0.8, 0.3, 0.5],
    )
    model = spec.compile()
    if cfg.elliptic_cone:
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = cfg.impratio
    g = model.actuator("actuator8").id
    model.actuator_gainprm[g, 0] *= cfg.gripper_stiffness_scale
    model.actuator_biasprm[g, 1] *= cfg.gripper_stiffness_scale
    # Keep the damping RATIO constant: zeta = kd / (2 sqrt(kp m)), so kd scales with sqrt(kp).
    # Scaling kp alone made the fingers underdamped; they slammed into the cube fast enough
    # to penetrate it (finger gap 14 mm on a 40 mm cube) and grasps failed. See devlog.md.
    model.actuator_biasprm[g, 2] *= np.sqrt(cfg.gripper_stiffness_scale)
    return model


class PickPlaceEnv:
    def __init__(self, cfg: SceneConfig | None = None):
        self.cfg = cfg or SceneConfig()
        self.model = build_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        self.arm_qpos = arm_qpos_indices(self.model)
        self.cube_qpos = self.model.joint("cube_joint").qposadr[0]   # 7 numbers: xyz + quat(wxyz)
        self.cube_body = self.model.body("cube").id
        self.target_site = self.model.site("target").id
        self.hand_body = self.model.body("hand").id
        self.episode: EpisodeSpec | None = None

    # ---------- episode setup ----------
    def sample_episode(self, seed: int) -> EpisodeSpec:
        rng = np.random.default_rng(seed)
        (cx0, cx1), (cy0, cy1) = self.cfg.cube_region
        (tx0, tx1), (ty0, ty1) = self.cfg.target_region
        while True:
            cube_xy = rng.uniform([cx0, cy0], [cx1, cy1])
            target_xy = rng.uniform([tx0, ty0], [tx1, ty1])
            if np.linalg.norm(cube_xy - target_xy) >= self.cfg.min_separation:
                break
        cube_yaw = rng.uniform(-np.pi, np.pi)
        return EpisodeSpec(seed, cube_xy, cube_yaw, target_xy)

    def reset(self, seed: int) -> EpisodeSpec:
        ep = self.sample_episode(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qpos] = Q_HOME
        d.qpos[[self.arm_qpos[-1] + 1, self.arm_qpos[-1] + 2]] = 0.04  # fingers open
        c = self.cube_qpos
        d.qpos[c:c + 3] = [*ep.cube_xy, self.cfg.cube_half_size]
        d.qpos[c + 3:c + 7] = [np.cos(ep.cube_yaw / 2), 0, 0, np.sin(ep.cube_yaw / 2)]
        m.site_pos[self.target_site, :2] = ep.target_xy
        d.ctrl[:7] = Q_HOME
        d.ctrl[7] = 255  # gripper open
        mujoco.mj_forward(m, d)
        self.episode = ep
        return ep

    # ---------- ground-truth observations (Stage 3 replaces these with vision) ----------
    def cube_pose(self) -> np.ndarray:
        return make_T(self.data.xmat[self.cube_body].reshape(3, 3), self.data.xpos[self.cube_body].copy())

    def cube_yaw(self) -> float:
        R = self.data.xmat[self.cube_body].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def target_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.target_site].copy()

    def tcp_pose(self) -> np.ndarray:
        T_hand = make_T(self.data.xmat[self.hand_body].reshape(3, 3), self.data.xpos[self.hand_body])
        return T_hand @ make_T(np.eye(3), np.array([0, 0, TCP_OFFSET]))

    def finger_width(self) -> float:
        i = self.arm_qpos[-1]
        return float(self.data.qpos[i + 1] + self.data.qpos[i + 2])

    # ---------- task outcome ----------
    def placement_error(self) -> float:
        return float(np.linalg.norm(self.data.xpos[self.cube_body][:2] - self.target_pos()[:2]))

    def cube_on_table(self) -> bool:
        return self.data.xpos[self.cube_body][2] < self.cfg.cube_half_size + 0.005

    def is_success(self) -> bool:
        return self.cube_on_table() and self.placement_error() < self.cfg.target_radius
