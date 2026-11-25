"""SO-101 pick-and-place scene: the Menagerie SO-101 + a cube + a target disc + a front camera.

Same conventions as envs/pick_place.py (seeded episodes, elliptic friction cone with
impratio 10, the cube's friction governing its contacts), sized for this arm: a 2.5 cm cube,
placed where the closed-form top-down IK can reach (`kinematics.py`, measured: 70-265 mm from
the pan axis at table level).

Joint control is the model's own: position servos with the STS3215 gains and the
+/-2.94 N m torque limit that the Menagerie model takes from LeRobot.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ari.so101.kinematics import JOINTS, TOOL_SITE, load_chain, measure_geometry, menagerie_path


@dataclass
class SO101Config:
    cube_half_size: float = 0.0125          # 2.5 cm cube (a wooden toy block)
    cube_mass: float = 0.020                # kg
    cube_friction: float = 1.0
    radius: tuple = (0.12, 0.23)            # m from the pan axis: inside the measured workspace with margin
    angle_deg: tuple = (-60.0, 60.0)        # around the pan axis, 0 = straight ahead
    min_separation: float = 0.08            # cube centre to target centre
    target_radius: float = 0.02
    elliptic_cone: bool = True
    impratio: float = 10.0


@dataclass
class SO101Episode:
    seed: int
    cube_xy: np.ndarray
    cube_yaw: float
    target_xy: np.ndarray


def build_model(cfg: SO101Config) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(menagerie_path() / "scene.xml"))
    cube = spec.worldbody.add_body(name="cube", pos=[0.2, 0.0, cfg.cube_half_size])
    cube.add_freejoint(name="cube_joint")
    cube.add_geom(name="cube_geom", type=mujoco.mjtGeom.mjGEOM_BOX, size=[cfg.cube_half_size] * 3,
                  mass=cfg.cube_mass, friction=[cfg.cube_friction, 0.005, 0.0001],
                  rgba=[0.85, 0.2, 0.2, 1.0], priority=1)
    spec.worldbody.add_site(name="target", pos=[0.2, 0.1, 0.0005], type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                            size=[cfg.target_radius, 0.0005, 0], rgba=[0.2, 0.8, 0.3, 0.5])
    from ari.perception.camera import look_at_quat

    cam_pos, cam_target = (0.42, -0.42, 0.36), (0.14, 0.0, 0.03)   # three-quarter view, arm not in the way
    spec.worldbody.add_camera(name="front_cam", pos=list(cam_pos), quat=list(look_at_quat(cam_pos, cam_target)), fovy=50.0)
    model = spec.compile()
    if cfg.elliptic_cone:
        model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.impratio = cfg.impratio
    return model


class SO101Env:
    def __init__(self, cfg: SO101Config | None = None):
        self.cfg = cfg or SO101Config()
        self.model = build_model(self.cfg)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.chain = load_chain(m)
        self.geometry = measure_geometry(self.chain)
        self.arm_qpos = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)] for j in JOINTS])
        self.gripper_qpos = m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "gripper")]
        self.arm_act = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in JOINTS])
        self.gripper_act = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper")
        self.cube_body = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cj = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_joint")
        self.cube_qpos, self.cube_dof = m.jnt_qposadr[cj], m.jnt_dofadr[cj]
        self.target_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "target")
        self.tool_site = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE)
        self.q_rest = np.array([0.0, -1.57, 1.57, 0.75, 0.0])       # folded, clear of the table

    def sample_episode(self, seed: int) -> SO101Episode:
        rng = np.random.default_rng(seed)
        c = self.cfg
        pan = self.geometry.pan_xy

        def point():
            r, a = rng.uniform(*c.radius), np.radians(rng.uniform(*c.angle_deg))
            return pan + r * np.array([np.cos(a), np.sin(a)])

        cube = point()
        while True:
            tgt = point()
            if np.linalg.norm(tgt - cube) >= c.min_separation:
                break
        return SO101Episode(seed, cube, float(rng.uniform(-np.pi / 4, np.pi / 4)), tgt)

    def reset(self, seed: int) -> SO101Episode:
        ep = self.sample_episode(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[self.arm_qpos] = self.q_rest
        d.ctrl[self.arm_act] = self.q_rest
        d.qpos[self.gripper_qpos] = d.ctrl[self.gripper_act] = 0.0
        h = self.cfg.cube_half_size
        d.qpos[self.cube_qpos:self.cube_qpos + 7] = [*ep.cube_xy, h, np.cos(ep.cube_yaw / 2), 0, 0, np.sin(ep.cube_yaw / 2)]
        m.site_pos[self.target_site][:2] = ep.target_xy
        mujoco.mj_forward(m, d)
        return ep

    # ---- reading the scene ----
    def cube_pos(self) -> np.ndarray:
        return self.data.xpos[self.cube_body].copy()

    def cube_yaw(self) -> float:
        R = self.data.xmat[self.cube_body].reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def target_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.target_site].copy()

    def tool_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.tool_site].copy()

    def q(self) -> np.ndarray:
        return self.data.qpos[self.arm_qpos].copy()

    def placement_error(self) -> float:
        return float(np.linalg.norm(self.cube_pos()[:2] - self.target_pos()[:2]))

    def is_success(self, tol: float = 0.015) -> bool:
        p = self.cube_pos()
        return self.placement_error() < tol and p[2] < self.cfg.cube_half_size + 0.003

    def step(self, n: int = 1) -> None:
        for _ in range(n):
            mujoco.mj_step(self.model, self.data)
