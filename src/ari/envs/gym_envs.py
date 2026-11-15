"""M16: the robot as Gymnasium environments for reinforcement learning.

An RL environment is defined by four things; all four are design decisions:

    observation  what the agent sees           (same vectors as Stage 4)
    action       what it can do                (same 20 Hz gripper deltas, scaled to [-1, 1])
    reward       what it is told is good       (below)
    termination  when an episode ends          (success, or a time limit)

Reward design. Each step:
    r = -0.01                        a small cost per step: faster is better
        + gamma * Phi(s') - Phi(s)   "potential-based shaping" (Ng, Harada & Russell, 1999)
        + bonus on success
Phi is a hand-written guess at how promising a state is (closer to the cube, cube closer to
the target, ...). Paying the CHANGE in Phi gives dense feedback, and this particular form
provably leaves the best policy unchanged: the shaping terms telescope, so an agent cannot
farm them by going in circles. Rewarding "distance to goal" directly has no such guarantee.

Time limits are TRUNCATION, not termination: the task did not end, we just stopped watching.
The learner must then still count the value of what would have come next (rl/ppo.py).
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

from ari.envs.pick_place import PickPlaceEnv
from ari.learning.demos import ActionExecutor, move_cube
from ari.learning.spaces import POLICY_DT, make_obs

MAX_STEP = 0.025       # m per decision at action = 1  (~0.25 m/s)
MAX_YAW_STEP = 0.15    # rad per decision at action = 1
ACTION_SCALE = np.array([MAX_STEP, MAX_STEP, MAX_STEP, MAX_YAW_STEP], dtype=np.float32)
RL_SEED_START = 30_000   # training episodes; evaluation stays on seeds 0-99


def to_physical(action: np.ndarray) -> np.ndarray:
    """Normalized 5-D action -> (dx, dy, dz, dyaw, gripper) in metres / radians / {0, 1}."""
    a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
    return np.array([*(a[:4] * ACTION_SCALE), float(a[4] > 0)], dtype=np.float32)


def to_normalized(physical: np.ndarray) -> np.ndarray:
    """Inverse of to_physical for the motion part; gripper becomes -1 / +1."""
    return np.array([*(physical[:4] / ACTION_SCALE), 1.0 if physical[4] > 0.5 else -1.0], dtype=np.float32)


class ReachEnv(gym.Env):
    """Move the gripper to a random point in space. The 'hello world' of robot RL."""

    metadata = {"render_modes": []}

    def __init__(self, gamma: float = 0.99, max_steps: int = 80, seed_start: int = RL_SEED_START):
        self.sim = PickPlaceEnv()
        self.gamma, self.max_steps = gamma, max_steps
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (9,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (3,), np.float32)
        self._next_seed = seed_start
        self.tolerance = 0.01

    def _obs(self):
        p = self.sim.tcp_pose()[:3, 3]
        return np.concatenate([p, self.goal, self.goal - p]).astype(np.float32)

    def _dist(self):
        return float(np.linalg.norm(self.goal - self.sim.tcp_pose()[:3, 3]))

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed, self._next_seed = self._next_seed, self._next_seed + 1
        self.sim.reset(seed)
        rng = np.random.default_rng(seed + 700_000)
        self.goal = rng.uniform([0.35, -0.30, 0.12], [0.65, 0.30, 0.45])
        self.exec = ActionExecutor(self.sim, joint_interp=True)
        self.t = 0
        self.phi = -2.0 * self._dist()
        return self._obs(), {"episode_seed": seed}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1, 1)
        self.exec.step(np.array([*(a * MAX_STEP), 0.0, 0.0], dtype=np.float32))
        self.t += 1
        d = self._dist()
        phi = -2.0 * d
        reward = -0.01 + self.gamma * phi - self.phi
        self.phi = phi
        terminated = d < self.tolerance
        if terminated:
            reward += 1.0
        truncated = (not terminated) and self.t >= self.max_steps
        return self._obs(), float(reward), terminated, truncated, {"is_success": terminated, "distance": d}


class PickPlaceGymEnv(gym.Env):
    """Pick the cube, place it on the target. Observation: the 20-D state vector of Stage 4."""

    metadata = {"render_modes": []}

    def __init__(self, gamma: float = 0.99, max_steps: int = 300, disturb_prob: float = 0.0,
                 seed_start: int = RL_SEED_START, joint_interp: bool = True, phi_offset: float = 0.0):
        """phi_offset: constant added to the potential. With gamma < 1, standing still earns
        (gamma - 1) * Phi per step: a small INCOME wherever Phi is negative, which roughly
        cancelled the time penalty and gave a from-scratch learner a place to sit (devlog.md).
        An offset of 2.5 makes Phi non-negative, so idling always costs. 0.0 reproduces the
        residual-RL results, which were trained before this was found."""
        self.phi_offset = phi_offset
        self.sim = PickPlaceEnv()
        self.gamma, self.max_steps, self.disturb_prob = gamma, max_steps, disturb_prob
        self.joint_interp = joint_interp
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (20,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
        self._next_seed = seed_start

    # -- reward ---------------------------------------------------------------------------
    def _placed(self) -> bool:
        return self.sim.is_success()

    def _potential(self) -> float:
        s = self.sim
        cube, tcp, tgt = s.data.xpos[s.cube_body], s.tcp_pose()[:3, 3], s.target_pos()
        half = s.cfg.cube_half_size
        d_ct = min(float(np.linalg.norm(cube[:2] - tgt[:2])), 0.6)
        d_gc = min(float(np.linalg.norm(tcp - cube)), 0.6)
        if self._placed():
            # Cube is home: what remains is letting go and backing away.
            return self.phi_offset + 0.3 + 0.2 * min(s.finger_width() / 0.08, 1.0) + min(max(tcp[2] - 0.02, 0.0), 0.13)
        lift = min(max(cube[2] - half, 0.0), 0.10) / 0.10
        return self.phi_offset - 2.0 * d_ct - 1.0 * d_gc + 0.3 * lift * min(1.0, d_ct / 0.05)

    def _done(self) -> bool:
        s = self.sim
        return s.is_success() and s.finger_width() > 0.07 and s.tcp_pose()[2, 3] > 0.06

    # -- gym API --------------------------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None:
            seed, self._next_seed = self._next_seed, self._next_seed + 1
        self.sim.reset(seed)
        self.exec = ActionExecutor(self.sim, joint_interp=self.joint_interp)
        self.t, self.hold, self.ever_lifted, self.closest = 0, 0, False, 1.0
        rng = np.random.default_rng(seed + 800_000)
        self._disturb = None
        if rng.random() < self.disturb_prob:
            from ari.learning.evaluate import CubeShift

            self._disturb = CubeShift(seed)
        self.phi = self._potential()
        return make_obs(self.sim), {"episode_seed": seed}

    def step(self, action):
        if self._disturb is not None:
            self._disturb(self.sim, self.t * POLICY_DT)
        self.exec.step(to_physical(action))
        self.t += 1
        phi = self._potential()
        reward = -0.01 + self.gamma * phi - self.phi
        self.phi = phi
        self.hold = self.hold + 1 if self._done() else 0
        terminated = self.hold >= 10            # held for 0.5 s, the Stage 4 success criterion
        if terminated:
            reward += 5.0
        truncated = (not terminated) and self.t >= self.max_steps
        s = self.sim
        self.ever_lifted = self.ever_lifted or bool(s.data.xpos[s.cube_body][2] > 0.06)
        self.closest = min(self.closest, float(np.linalg.norm(s.tcp_pose()[:3, 3] - s.data.xpos[s.cube_body])))
        info = {"is_success": terminated, "lifted": self.ever_lifted, "closest_to_cube": self.closest}
        return make_obs(self.sim), float(reward), terminated, truncated, info


__all__ = ["ReachEnv", "PickPlaceGymEnv", "to_physical", "to_normalized", "move_cube"]
