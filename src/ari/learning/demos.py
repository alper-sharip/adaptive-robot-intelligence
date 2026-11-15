"""M13: record demonstrations from the scripted expert, and roll out learned policies.

File format: one .npz with flat arrays, one row per policy step. Column names follow the
LeRobot dataset convention (observation.state, action, episode_index, frame_index,
timestamp) so that converting to a LeRobotDataset in M15 is a rename, not a redesign.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from ari.control.cartesian import CartesianController
from ari.envs.pick_place import PickPlaceEnv
from ari.learning.spaces import (GRIPPER_DOWN, POLICY_DT, action_from_target, make_obs,
                                 target_from_action, tcp_yaw, wrap)
from ari.robot.transforms import make_T, rot_z
from ari.skills.pick_place import ScriptedPickPlace

TRAIN_SEED_START = 10_000   # evaluation uses seeds 0-99; demonstrations never overlap


class ActionExecutor:
    """Turns one 20 Hz policy action into smooth 100 Hz controller commands.

    The command is interpolated linearly from the previous target to the new one across the
    policy step. Jumping straight to each new target (a 'staircase') jolts the stiff joint
    servos 20 times a second; with Menagerie's ~1 N grip the cube slipped out during the
    lift and the EXPERT failed 2 of 10 episodes through this interface (devlog.md).
    Both demonstration recording and policy rollouts go through this one class.
    """

    def __init__(self, env: PickPlaceEnv, joint_interp: bool = False):
        """joint_interp=False: interpolate the Cartesian target, IK at every 100 Hz tick
        (Stage 4 behaviour). joint_interp=True: IK once per decision, interpolate the joint
        command. 5x fewer IK solves; used for reinforcement learning, where simulation
        speed decides what is feasible."""
        self.env = env
        self.ctrl = CartesianController(env.model, env.data)
        # Start the commanded posture at the MEASURED one (identical to home right after a
        # reset; matters when an executor is created mid-episode, as the task planner does).
        self.ctrl.reset(env.data.qpos[env.arm_qpos].copy())
        self.ctrl.gripper_cmd = float(env.data.ctrl[7])
        self.every = int(round(POLICY_DT / self.ctrl.control_dt))
        self.prev = env.tcp_pose().copy()
        self.joint_interp = joint_interp

    def step(self, action: np.ndarray) -> None:
        T_new, closed = target_from_action(self.env.tcp_pose(), action)
        self.ctrl.set_gripper(closed)
        if self.joint_interp:
            q0 = self.ctrl.q_cmd.copy()
            self.ctrl.set_pose_target(T_new)
            q1 = self.ctrl.q_cmd.copy()
            for i in range(1, self.every + 1):
                self.ctrl.set_joint_target(q0 + (q1 - q0) * i / self.every)
                self.ctrl.step()
            self.prev = T_new
            return
        y0, y1 = tcp_yaw(self.prev), tcp_yaw(T_new)
        for i in range(1, self.every + 1):
            f = i / self.every
            T = make_T(rot_z(y0 + f * wrap(y1 - y0)) @ GRIPPER_DOWN,
                       (1 - f) * self.prev[:3, 3] + f * T_new[:3, 3])
            self.ctrl.set_pose_target(T)
            self.ctrl.step()
        self.prev = T_new


def record_episode(env: PickPlaceEnv, seed: int, expert=None):
    """Run the expert on one seeded episode. Returns (obs [T,20], act [T,5], success)."""
    expert = expert or ScriptedPickPlace()
    env.reset(seed)
    ex = ActionExecutor(env)
    expert.reset(env)
    dt = ex.ctrl.control_dt
    obs, act = [], []
    # The expert acts through the SAME interface the learner will use: one decision per
    # policy step (20 Hz), executed by ActionExecutor. Recording a 100 Hz expert and replaying
    # it at 20 Hz does not reproduce the episode: actions are relative to the measured pose,
    # so the small timing difference accumulates (devlog.md).
    while True:
        o, T_now = make_obs(env), env.tcp_pose().copy()
        a = None
        for _ in range(ex.every):                   # advance the expert's clock one policy step
            a = expert.act(env, dt, q_cmd=ex.ctrl.q_cmd)
            if a is None:
                break
        if a is None:
            break
        action = action_from_target(T_now, a[0], a[1])
        ex.step(action)
        obs.append(o)
        act.append(action)
    for _ in range(int(0.3 / dt)):
        ex.ctrl.step()
    return np.array(obs), np.array(act), env.is_success()


def record_reactive(env: PickPlaceEnv, seed: int, expert, max_time: float = 25.0):
    """Record a state-feedback expert (a callable env -> action). Returns (obs, act, success)."""
    obs, act = [], []

    class Recorder:
        def __call__(self, o):
            a = expert(env)
            obs.append(o), act.append(a)
            return a

    res = rollout(env, Recorder(), seed, max_time=max_time)
    return np.array(obs), np.array(act), res.success


def collect(n: int, start_seed: int = TRAIN_SEED_START, env: PickPlaceEnv | None = None, expert_factory=None):
    """Collect n SUCCESSFUL demonstrations (failed expert episodes are skipped and counted)."""
    env = env or PickPlaceEnv()
    O, A, E, F, S = [], [], [], [], []
    seed, skipped = start_seed, 0
    while len(S) < n:
        o, a, ok = record_episode(env, seed, expert_factory() if expert_factory else None)
        if ok:
            ep = len(S)
            O.append(o), A.append(a)
            E.append(np.full(len(o), ep)), F.append(np.arange(len(o)))
            S.append(seed)
        else:
            skipped += 1
        seed += 1
    return {
        "observation.state": np.concatenate(O), "action": np.concatenate(A),
        "episode_index": np.concatenate(E), "frame_index": np.concatenate(F),
        "timestamp": np.concatenate(F) * POLICY_DT, "episode_seed": np.array(S),
        "skipped_failed_expert_episodes": np.array(skipped),
    }


def save(data: dict, path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load(path: Path, n_episodes: int | None = None) -> dict:
    d = dict(np.load(path))
    if n_episodes is not None:
        keep = d["episode_index"] < n_episodes
        for key in ("observation.state", "action", "episode_index", "frame_index", "timestamp"):
            d[key] = d[key][keep]
        d["episode_seed"] = d["episode_seed"][:n_episodes]
    return d


# ---------------------------------------------------------------- rollouts
@dataclass
class RolloutResult:
    seed: int
    success: bool
    failure: str
    placement_error: float
    duration: float
    max_cube_height: float


def run_policy(env: PickPlaceEnv, ex: "ActionExecutor", policy, max_time: float = 25.0, disturbance=None,
               frame_callback=None, obs_fn=make_obs) -> tuple[bool, float]:
    """Run a policy from the CURRENT state until the task is done or time runs out.
    Returns (success, highest cube height reached). Used by rollout() and by the task
    planner (Stage 6), which runs several skills one after another in the same episode.

    Success = cube resting in the target zone, gripper open and raised, held for 0.5 s.
    """
    max_z, ok_time = 0.0, 0.0
    t0 = env.data.time
    while env.data.time - t0 < max_time:
        if disturbance is not None:
            disturbance(env, env.data.time - t0)
        ex.step(policy(obs_fn(env)))
        if frame_callback is not None:
            frame_callback(env)
        max_z = max(max_z, float(env.data.xpos[env.cube_body][2]))
        done = env.is_success() and env.finger_width() > 0.07 and env.tcp_pose()[2, 3] > 0.06
        ok_time = ok_time + POLICY_DT if done else 0.0
        if ok_time >= 0.5:
            return True, max_z
    return False, max_z


def rollout(env: PickPlaceEnv, policy, seed: int, max_time: float = 25.0, disturbance=None,
            frame_callback=None, joint_interp: bool = False) -> RolloutResult:
    """Run a learned (or any obs->action) policy closed-loop for one seeded episode.

    policy: callable obs -> action (5,), with an optional .reset().
    disturbance: optional callable (env, t) applied every policy step (e.g. move the cube).
    """
    env.reset(seed)
    ex = ActionExecutor(env, joint_interp=joint_interp)
    if hasattr(policy, "reset"):
        policy.reset()
    half = env.cfg.cube_half_size
    t0 = env.data.time
    success, max_z = run_policy(env, ex, policy, max_time, disturbance, frame_callback)
    if success:
        return RolloutResult(seed, True, "", env.placement_error(), env.data.time - t0, max_z)
    if max_z < half + 0.03:
        failure = "never_lifted"
    elif not env.cube_on_table():
        failure = "still_holding_or_off_table"
    elif env.placement_error() >= env.cfg.target_radius:
        failure = "dropped_or_misplaced"
    else:
        failure = "placed_but_did_not_retreat"
    return RolloutResult(seed, False, failure, env.placement_error(), env.data.time - t0, max_z)


def move_cube(env: PickPlaceEnv, xy) -> None:
    """Teleport the cube on the table (used for disturbance tests)."""
    c = env.cube_qpos
    env.data.qpos[c:c + 2] = xy
    env.data.qpos[c + 2] = env.cfg.cube_half_size
    dof = getattr(env, "cube_dofadr", None)
    if dof is None:
        dof = env.model.joint("cube_joint").dofadr[0]
    env.data.qvel[dof:dof + 6] = 0
    mujoco.mj_forward(env.model, env.data)
