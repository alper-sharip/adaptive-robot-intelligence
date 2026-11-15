"""M8: collision checking for motion planning.

Planning needs to ask "is joint configuration q safe?" thousands of times, without
running physics. MuJoCo can answer that directly:

    set qpos -> mj_kinematics (body poses) -> geometry queries

using a *private* MjData, so planning never disturbs the running simulation.

Two kinds of check, because they need different rules:
  1. Robot (+ held object) vs OBSTACLES: must stay at least `margin` away. We ask for the
     exact signed distance between each pair of shapes with mj_geomDistance.
  2. Self-collision and the floor: only actual overlap counts. Links 5 and 7 always sit
     within 2 cm of each other, so a margin there would make every pose "unsafe".
     These come from mj_collision, which reports touching/overlapping shapes.

(First version inflated geom_margin for everything and called mj_collision. That was wrong
in both directions: the effective margin depended on the shape types involved, so a hand
3.7 cm from the floor was flagged while a fingertip 7 mm away was not. See devlog.md.)

The held object: once the cube is grasped it moves with the hand, so for planning it is
part of the robot. Its pose relative to the hand is MEASURED when planning starts.
"""

from __future__ import annotations

import mujoco
import numpy as np

from ari.robot.model import arm_qpos_indices
from ari.robot.transforms import make_T

GRIPPER_BODIES = ("hand", "left_finger", "right_finger")


def _rot_to_quat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


class CollisionChecker:
    def __init__(self, model: mujoco.MjModel, obstacle_bodies=("wall",), margin: float = 0.02):
        self.model = model
        self.data = mujoco.MjData(model)  # private scratch copy
        self.margin = margin
        self.arm_qpos = arm_qpos_indices(model)
        names = [model.body(b).name for b in range(model.nbody)]
        first, last = names.index("link0"), names.index("right_finger")
        self.robot_bodies = set(range(first, last + 1))
        self.gripper_bodies = {model.body(n).id for n in GRIPPER_BODIES}
        obstacle_ids = {model.body(n).id for n in obstacle_bodies if n in names}
        colliding = lambda g: model.geom_contype[g] or model.geom_conaffinity[g]  # noqa: E731
        self.obstacle_geoms = [g for g in range(model.ngeom)
                               if model.geom_bodyid[g] in obstacle_ids and colliding(g)]
        self.robot_geoms = [g for g in range(model.ngeom)
                            if model.geom_bodyid[g] in self.robot_bodies and colliding(g)]
        self.held_body = None
        self.held_geoms: list[int] = []
        self.T_hand_obj = None
        self.check_count = 0

    # ---------- held object ----------
    def attach(self, body_name: str, sim_data: mujoco.MjData) -> None:
        """Treat `body_name` as rigidly attached to the hand, at its CURRENT relative pose."""
        m = self.model
        b, h = m.body(body_name).id, m.body("hand").id
        T_world_hand = make_T(sim_data.xmat[h].reshape(3, 3), sim_data.xpos[h])
        T_world_obj = make_T(sim_data.xmat[b].reshape(3, 3), sim_data.xpos[b])
        self.held_body = b
        self.held_geoms = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
        self.T_hand_obj = np.linalg.inv(T_world_hand) @ T_world_obj
        self.data.qpos[:] = sim_data.qpos  # start from the real state (fingers, other objects)

    def detach(self) -> None:
        self.held_body, self.held_geoms, self.T_hand_obj = None, [], None

    def sync(self, sim_data: mujoco.MjData) -> None:
        """Copy the full simulator state (e.g. object positions) into the private data."""
        self.data.qpos[:] = sim_data.qpos

    # ---------- queries ----------
    def _set(self, q: np.ndarray) -> None:
        m, d = self.model, self.data
        d.qpos[self.arm_qpos] = q
        mujoco.mj_kinematics(m, d)
        if self.held_body is not None:
            h = m.body("hand").id
            T = make_T(d.xmat[h].reshape(3, 3), d.xpos[h]) @ self.T_hand_obj
            adr = m.jnt_qposadr[m.body_jntadr[self.held_body]]
            d.qpos[adr:adr + 3] = T[:3, 3]
            d.qpos[adr + 3:adr + 7] = _rot_to_quat(T[:3, :3])
            mujoco.mj_kinematics(m, d)

    def collisions(self, q: np.ndarray, first_only: bool = False) -> list[tuple[str, str, float]]:
        """Offending pairs at configuration q: (body1, body2, distance)."""
        m, d = self.model, self.data
        self._set(q)
        self.check_count += 1
        bad = []

        # 1. Clearance from obstacles (exact distances, with margin).
        for g1 in self.robot_geoms + self.held_geoms:
            for g2 in self.obstacle_geoms:
                dist = mujoco.mj_geomDistance(m, d, g1, g2, self.margin, None)
                if dist < self.margin:
                    bad.append((m.body(m.geom_bodyid[g1]).name, m.body(m.geom_bodyid[g2]).name, dist))
                    if first_only:
                        return bad

        # 2. Actual overlap: self-collision, floor, other objects.
        mujoco.mj_collision(m, d)
        moving = self.robot_bodies | ({self.held_body} if self.held_body is not None else set())
        for c in d.contact[: d.ncon]:
            if c.dist >= 0:
                continue
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            pair = {b1, b2}
            if not pair & moving:
                continue  # e.g. a different object resting on the table
            if self.held_body in pair and pair & self.gripper_bodies:
                continue  # the fingers are supposed to touch the held cube
            if pair <= self.gripper_bodies:
                continue
            bad.append((m.body(b1).name, m.body(b2).name, float(c.dist)))
            if first_only:
                return bad
        return bad

    def is_free(self, q: np.ndarray) -> bool:
        return not self.collisions(q, first_only=True)

    def segment_free(self, q0: np.ndarray, q1: np.ndarray, resolution: float = 0.03) -> bool:
        """Check the straight joint-space segment q0 -> q1, samples at most `resolution` rad apart.

        This is sampling, not proof: a thin obstacle could slip between samples. The
        2 cm margin is what makes 0.03 rad acceptable (the hand moves a few cm at most
        per step only for the base joints at full reach; usually far less).
        """
        n = max(1, int(np.ceil(np.abs(q1 - q0).max() / resolution)))
        for i in range(1, n + 1):
            if not self.is_free(q0 + (q1 - q0) * i / n):
                return False
        return True
