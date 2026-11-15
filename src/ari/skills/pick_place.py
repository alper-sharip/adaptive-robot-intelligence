"""M7: scripted pick-and-place, the classical baseline.

The policy is a fixed sequence of phases. Each phase is a min-jerk move to a goal
pose (or a pause) with the gripper open or closed. It reads the cube pose from the
simulator's ground truth; Stage 3 swaps that for camera-based estimates.

Policy interface (shared by every future policy, scripted or learned):
    policy.reset(env)
    action = policy.act(env)   ->  (T_tcp_target 4x4, gripper_closed bool), or None when finished
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ari.control.cartesian import CartesianController
from ari.envs.pick_place import PickPlaceEnv
from ari.planning.trajectory import PoseTrajectory, duration_for
from ari.robot.transforms import make_T, rot_x, rot_z

GRIPPER_DOWN = rot_x(np.pi)  # hand z-axis pointing at the table


def grasp_yaw(cube_yaw: float) -> float:
    """A cube looks the same every 90 deg, so pick the equivalent yaw in [-45, 45] deg.
    This keeps wrist rotations small. Fingers close along the hand's y-axis, which
    lines up with a cube face normal when the hand yaw equals the cube yaw (mod 90 deg)."""
    return (cube_yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


@dataclass
class Phase:
    name: str
    goal: np.ndarray            # 4x4 TCP pose at the end of the phase
    gripper_closed: bool
    hold: float = 0.0           # extra time to wait at the goal (e.g. for the gripper to close)


@dataclass
class ScriptedPickPlace:
    approach_height: float = 0.12    # above the cube centre before descending
    grasp_depth: float = 0.0         # TCP height relative to the cube centre when grasping
    carry_height: float = 0.15       # TCP height while transporting
    place_clearance: float = 0.005   # cube bottom height above the table when releasing
    close_time: float = 0.6          # s to wait for the fingers to close
    open_time: float = 0.4
    max_speed: float = 0.25          # m/s peak TCP speed
    phases: list = field(default_factory=list)

    def reset(self, env: PickPlaceEnv) -> None:
        cube = env.cube_pose()[:3, 3]
        target = env.target_pos()
        half = env.cfg.cube_half_size
        R = rot_z(grasp_yaw(env.cube_yaw())) @ GRIPPER_DOWN

        def pose(xyz):
            return make_T(R, np.asarray(xyz, dtype=float))

        place_z = half + self.place_clearance + self.grasp_depth
        self.phases = [
            Phase("approach", pose(cube + [0, 0, self.approach_height]), False),
            Phase("descend", pose(cube + [0, 0, self.grasp_depth]), False),
            Phase("grasp", pose(cube + [0, 0, self.grasp_depth]), True, hold=self.close_time),
            Phase("lift", pose([*cube[:2], self.carry_height]), True),
            Phase("transport", pose([*target[:2], self.carry_height]), True),
            Phase("lower", pose([*target[:2], place_z]), True),
            Phase("release", pose([*target[:2], place_z]), False, hold=self.open_time),
            Phase("retreat", pose([*target[:2], self.carry_height]), False),
        ]
        self._i = 0
        self._t = 0.0
        self._start = env.tcp_pose()
        self._traj = self._make_traj(self._start, self.phases[0])

    def _make_traj(self, T0, phase):
        return PoseTrajectory(T0, phase.goal, duration_for(T0, phase.goal, max_speed=self.max_speed))

    @property
    def phase(self) -> str:
        return self.phases[self._i].name if self._i < len(self.phases) else "done"

    def act(self, env: PickPlaceEnv, dt: float):
        if self._i >= len(self.phases):
            return None
        ph = self.phases[self._i]
        T = self._traj(self._t)
        self._t += dt
        if self._t >= self._traj.duration + ph.hold:
            self._i += 1
            self._t = 0.0
            if self._i < len(self.phases):
                self._traj = self._make_traj(ph.goal, self.phases[self._i])
        return T, ph.gripper_closed


@dataclass
class EpisodeResult:
    seed: int
    success: bool
    failure: str            # "" if success, else which check failed
    placement_error: float  # m
    duration: float         # s of simulated time
    ik_failures: int
    slip: float = 0.0       # m the cube slid down in the fingers while carried (lift end -> transport end)
    phase_log: list = field(default_factory=list)


def run_episode(env: PickPlaceEnv, policy, seed: int, frame_callback=None, frame_every: int = 4) -> EpisodeResult:
    """Run one seeded episode and diagnose what went wrong, if anything."""
    env.reset(seed)
    ctrl = CartesianController(env.model, env.data)
    ctrl.reset()
    policy.reset(env)
    failure = ""
    prev_phase = policy.phase
    phase_log = []
    offset_after_lift = None
    slip = 0.0
    k = 0
    while True:
        action = policy.act(env, ctrl.control_dt)
        if action is None:
            break
        T, closed = action
        ctrl.set_pose_target(T)
        ctrl.set_gripper(closed)
        ctrl.step()
        if frame_callback is not None and k % frame_every == 0:
            frame_callback(env, policy.phase)
        k += 1

        # Checks at phase boundaries: these turn "it failed" into "it failed HERE".
        if policy.phase != prev_phase:
            phase_log.append((prev_phase, round(env.data.time, 3)))
            cube_z = env.data.xpos[env.cube_body][2]
            offset = env.tcp_pose()[2, 3] - cube_z   # how far the cube hangs below the TCP
            if prev_phase == "lift":
                offset_after_lift = offset
                if cube_z < env.cfg.cube_half_size + 0.03:
                    failure = "grasp_failed"
            elif prev_phase == "transport" and not failure:
                slip = offset - offset_after_lift
                gap = np.linalg.norm(env.data.xpos[env.cube_body] - env.tcp_pose()[:3, 3])
                if gap > 0.03:
                    failure = "dropped_in_transport"
            prev_phase = policy.phase

    # Let the cube settle after release.
    for _ in range(int(0.3 / ctrl.control_dt)):
        ctrl.step()

    success = env.is_success()
    if not success and not failure:
        failure = "placement_miss" if env.cube_on_table() else "not_on_table"
    return EpisodeResult(seed, success, "" if success else failure, env.placement_error(),
                         float(env.data.time), ctrl.ik_failures, slip, phase_log)
