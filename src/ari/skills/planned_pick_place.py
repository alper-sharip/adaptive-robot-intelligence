"""M8: pick-and-place that plans around obstacles.

Same phases as the Stage 2 scripted policy, but every motion is collision-checked and
executed as a JOINT trajectory, so the arm posture is chosen, not left to chance:

  approach  : RRT-Connect from the current posture to a collision-free approach posture
  descend   : straight-line Cartesian move, converted to a dense joint path (posture kept)
  grasp     : close the gripper
  lift      : straight-line up, dense joint path
  transport : RRT-Connect, with the cube ATTACHED to the hand for collision checking
  lower     : straight-line down, dense joint path
  release   : open the gripper
  retreat   : straight-line up, dense joint path

Pose selection uses the cube's 90-degree symmetry: four hand yaws are equally valid for
grasping and for placing; we take the first whose whole sub-plan is collision-free.

Why not keep using the Cartesian controller's IK for the straight moves? Its null-space
term pulls the elbow toward the home posture; near the wall that drags link 5 into it
(found during development, see devlog.md). Precomputed joint paths keep the posture we
verified.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from ari.envs.pick_place import PickPlaceEnv
from ari.planning.collision import CollisionChecker
from ari.planning.redundancy import anchored_paths, cartesian_path
from ari.planning.rrt import path_length, rrt_connect, shortcut
from ari.planning.trajectory import DenseJointPath, JointPathTrajectory, duration_for
from ari.robot.model import arm_qpos_indices
from ari.robot.transforms import make_T, rot_z
from ari.skills.pick_place import GRIPPER_DOWN, grasp_yaw

YAW_OFFSETS = (0.0, np.pi / 2, -np.pi / 2, np.pi)  # equivalent for a cube; ordered by wrist rotation


def _hold(q, duration):
    return DenseJointPath([q, q], duration)


@dataclass
class PlannedPickPlace:
    approach_height: float = 0.12
    grasp_depth: float = 0.0
    carry_height: float = 0.15
    place_clearance: float = 0.005
    close_time: float = 0.6
    open_time: float = 0.4
    max_speed: float = 0.25          # m/s, straight-line moves
    max_joint_speed: float = 0.8     # rad/s, planned free-space moves
    margin: float = 0.02             # m clearance from obstacles
    plan_seed: int = 0
    plan_info: dict = field(default_factory=dict)

    # ---------- planning ----------
    def reset(self, env: PickPlaceEnv) -> None:
        t0 = time.time()
        self.env = env
        self.cc = CollisionChecker(env.model, margin=self.margin)
        self.cc.sync(env.data)
        self.plan_info = {}
        self.failure_reason = ""
        self.half = env.cfg.cube_half_size
        q0 = env.data.qpos[arm_qpos_indices(env.model)].copy()
        cube = env.cube_pose()[:3, 3]

        self._grasp = None
        for k, off in enumerate(YAW_OFFSETS):
            R = rot_z(grasp_yaw(env.cube_yaw()) + off) @ GRIPPER_DOWN
            T_app = make_T(R, cube + [0, 0, self.approach_height])
            T_grasp = make_T(R, cube + [0, 0, self.grasp_depth])
            T_lift = make_T(R, np.array([*cube[:2], self.carry_height]))
            found = anchored_paths(T_grasp, [T_app, T_lift], self.cc, q_seed=q0, seed=self.plan_seed + k)
            if found is None:
                continue
            _, (to_app, up) = found
            down = to_app[::-1]  # traced upward from the grasp, executed downward
            self._grasp = dict(R=R, T_app=T_app, T_grasp=T_grasp, T_lift=T_lift, down=down, up=up)
            self.plan_info["grasp_yaw_candidate"] = k
            break

        self.phases = ["approach", "descend", "grasp", "lift", "transport", "lower", "release", "retreat"]
        self._i, self._t = 0, 0.0
        self._traj = None
        if self._grasp is None:
            self.failure_reason = "no_grasp_plan"
        else:
            self._traj = self._build("approach", q0)
        self.plan_time = time.time() - t0

    def _rrt(self, q_start, q_goal):
        res = rrt_connect(self.cc, q_start, q_goal, seed=self.plan_seed)
        if res.path is None:
            return None
        path = shortcut(self.cc, res.path, seed=self.plan_seed)
        self.plan_info.setdefault("rrt", []).append(
            dict(iterations=res.iterations, checks=res.checks, raw_waypoints=len(res.path),
                 waypoints=len(path), raw_length=round(path_length(res.path), 3),
                 length=round(path_length(path), 3)))
        return path

    def _build(self, name, q_cmd):
        """Plan the trajectory for phase `name`, starting from commanded posture q_cmd."""
        g, env = self._grasp, self.env
        if name == "approach":
            path = self._rrt(q_cmd, g["down"][0])
            if path is None:
                self.failure_reason = "no_approach_path"
                return None
            return JointPathTrajectory(path, self.max_joint_speed)
        if name == "descend":
            return DenseJointPath(g["down"], duration_for(g["T_app"], g["T_grasp"], self.max_speed))
        if name == "grasp":
            return _hold(g["down"][-1], self.close_time)
        if name == "lift":
            return DenseJointPath(g["up"], duration_for(g["T_grasp"], g["T_lift"], self.max_speed))
        if name == "transport":
            return self._plan_transport(q_cmd)
        if name == "lower":
            p = self._place
            return DenseJointPath(p["down"], duration_for(p["T_carry"], p["T_place"], self.max_speed))
        if name == "release":
            return _hold(self._place["down"][-1], self.open_time)
        if name == "retreat":
            self.cc.detach()
            self.cc.sync(env.data)  # the cube now sits at the target
            T0 = self._place["T_place"]
            T1 = make_T(T0[:3, :3], np.array([*T0[:2, 3], self.carry_height]))
            up = cartesian_path(T0, T1, q_cmd, self.cc)
            if up is None:
                # Open fingers are wider than closed ones; fall back to the reverse of the
                # lowering path, which was checked with the cube in hand.
                up = self._place["down"][::-1]
            return DenseJointPath(up, duration_for(T0, T1, self.max_speed))
        raise ValueError(name)

    def _plan_transport(self, q_cmd):
        env = self.env
        self.cc.attach("cube", env.data)  # measured cube-in-hand pose, not assumed
        target = env.target_pos()
        place_z = self.half + self.place_clearance + self.grasp_depth
        Rg = self._grasp["R"] @ GRIPPER_DOWN.T          # = rot_z(grasp yaw)
        yaw0 = float(np.arctan2(Rg[1, 0], Rg[0, 0]))
        self._place = None
        for k, off in enumerate(YAW_OFFSETS):
            R = rot_z(yaw0 + off) @ GRIPPER_DOWN
            T_carry = make_T(R, np.array([*target[:2], self.carry_height]))
            T_place = make_T(R, np.array([*target[:2], place_z]))
            found = anchored_paths(T_place, [T_carry], self.cc, q_seed=q_cmd, seed=self.plan_seed + 10 + k)
            if found is None:
                continue
            down = found[1][0][::-1]
            self._place = dict(T_carry=T_carry, T_place=T_place, down=down)
            self.plan_info["place_yaw_candidate"] = k
            break
        if self._place is None:
            self.failure_reason = "no_place_plan"
            return None
        path = self._rrt(q_cmd, self._place["down"][0])
        if path is None:
            self.failure_reason = "no_transport_path"
            return None
        return JointPathTrajectory(path, self.max_joint_speed)

    # ---------- execution (same interface as ScriptedPickPlace) ----------
    @property
    def phase(self) -> str:
        return self.phases[self._i] if self._i < len(self.phases) else "done"

    def act(self, env: PickPlaceEnv, dt: float, q_cmd=None):
        if self._i >= len(self.phases) or self._traj is None:
            return None
        name = self.phases[self._i]
        q = self._traj(self._t)
        self._t += dt
        if self._traj.done(self._t):
            self._i += 1
            self._t = 0.0
            if self._i < len(self.phases):
                t0 = time.time()
                self._traj = self._build(self.phases[self._i], q)
                self.plan_time += time.time() - t0
        closed = name in ("grasp", "lift", "transport", "lower")
        return q, closed
