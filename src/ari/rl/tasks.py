"""Task builders for PPO (importable functions, because worker processes rebuild them), the
residual policy (M18), and checkpoint save/load.

Residual reinforcement learning (Johannink et al., 2019; Silver et al., 2018):

    action = base_policy(obs) + small learned correction(obs)

The base is the cloned policy from Stage 4, frozen. The correction network starts at exactly
zero, so training begins from the clone's behaviour instead of from random flailing, and RL
only has to discover millimetre-scale fixes. Imitation supplies the skill; the reward
supplies what imitation could not: feedback on whether the task actually succeeded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ari.rl.ppo import GaussianPolicy, mlp


# ---------------------------------------------------------------- from-scratch tasks
def build_gym(worker_id: int, env_id: str, seed: int = 0, **_):
    """A standard Gymnasium benchmark, used to check the PPO implementation itself."""
    import gymnasium as gym

    env = gym.make(env_id)
    env.reset(seed=seed * 1000 + worker_id + 1)

    class Scaled(GaussianPolicy):
        def to_env(self, obs, raw):
            return (raw.clamp(-1, 1) * float(env.action_space.high[0])).numpy()

    return env, Scaled(env.observation_space.shape[0], env.action_space.shape[0])


def build_reach(worker_id: int, seed: int = 0, **_):
    from ari.envs.gym_envs import RL_SEED_START, ReachEnv

    env = ReachEnv(seed_start=RL_SEED_START + 100_000 * (worker_id + 1) + 1_000_000 * seed)
    return env, GaussianPolicy(9, 3)


def build_pick(worker_id: int, seed: int = 0, disturb_prob: float = 0.0, phi_offset: float = 0.0, **_):
    from ari.envs.gym_envs import RL_SEED_START, PickPlaceGymEnv

    env = PickPlaceGymEnv(disturb_prob=disturb_prob, phi_offset=phi_offset,
                          seed_start=RL_SEED_START + 100_000 * (worker_id + 1) + 1_000_000 * seed)
    return env, GaussianPolicy(20, 5)


# ---------------------------------------------------------------- residual on a cloned policy
class ResidualPolicy(GaussianPolicy):
    """Frozen behaviour-cloned base + a learned correction that starts at zero."""

    def __init__(self, base_ckpt: dict, motion_scale: float = 0.2, gripper_scale: float = 2.0,
                 log_std: float = -1.2, hidden: int = 128):
        from ari.learning.bc import MLP

        super().__init__(base_ckpt["obs_dim"], 5, hidden=hidden, log_std=log_std,
                         fixed_norm=(base_ckpt["obs_mean"], base_ckpt["obs_std"]))
        self.actor = mlp([base_ckpt["obs_dim"], hidden, hidden, 5], zero_last=True)   # correction = 0 at start
        self.base = MLP(base_ckpt["obs_dim"], 5, base_ckpt["hidden"], base_ckpt["layers"])
        self.base.load_state_dict(base_ckpt["state_dict"])
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.register_buffer("act_mean", base_ckpt["act_mean"].clone())
        self.register_buffer("act_std", base_ckpt["act_std"].clone())
        # motion_scale: a correction of 1.0 moves the gripper by this fraction of a full step
        # (0.2 * 25 mm = 5 mm). The exploration noise (std ~0.3) is therefore ~1.5 mm.
        self.motion_scale, self.gripper_scale = motion_scale, gripper_scale
        self.base_ckpt = base_ckpt

    @torch.no_grad()
    def to_env(self, obs, raw):
        from ari.envs.gym_envs import ACTION_SCALE

        out = self.base(self._n(obs))
        motion = (out[:, :4] * self.act_std + self.act_mean) / torch.as_tensor(ACTION_SCALE)   # normalized units
        motion = motion + self.motion_scale * raw[:, :4]
        grip = out[:, 4:5] + self.gripper_scale * raw[:, 4:5]                                  # logit + correction
        return torch.cat([motion.clamp(-1, 1), torch.where(grip > 0, 1.0, -1.0)], dim=1).numpy()


def build_residual(worker_id: int, base_path: str, seed: int = 0, disturb_prob: float = 0.0, **_):
    from ari.envs.gym_envs import RL_SEED_START, PickPlaceGymEnv

    env = PickPlaceGymEnv(disturb_prob=disturb_prob,
                          seed_start=RL_SEED_START + 100_000 * (worker_id + 1) + 1_000_000 * seed)
    return env, ResidualPolicy(torch.load(base_path, weights_only=True))


# ---------------------------------------------------------------- checkpoints
def save(path: Path, policy: GaussianPolicy, kind: str, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    ck = {"kind": kind, "state_dict": policy.state_dict(), **(extra or {})}
    if kind == "residual":
        ck["base_ckpt"] = policy.base_ckpt
    torch.save(ck, path)


def load(path: Path) -> GaussianPolicy:
    ck = torch.load(path, weights_only=True)
    if ck["kind"] == "residual":
        policy = ResidualPolicy(ck["base_ckpt"])
    elif ck["kind"] == "ppo_pick":
        policy = GaussianPolicy(20, 5)
    elif ck["kind"] == "ppo_reach":
        policy = GaussianPolicy(9, 3)
    else:
        raise ValueError(ck["kind"])
    policy.load_state_dict(ck["state_dict"])
    return policy.eval()


class AsRolloutPolicy:
    """A PPO policy as the `obs -> physical action` callable that demos.rollout expects.
    Deterministic by default (the mean action); noisy=True keeps the exploration noise."""

    def __init__(self, policy: GaussianPolicy, noisy: bool = False):
        self.policy, self.noisy = policy, noisy

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        from ari.envs.gym_envs import to_physical

        return to_physical(self.policy.act(obs, deterministic=not self.noisy)[0])


def evaluate_reach(policy: GaussianPolicy | None, seeds=range(100)) -> dict:
    """Deterministic reach evaluation on fixed seeds. policy=None runs the classical answer:
    a proportional controller that steps straight at the goal."""
    from ari.envs.gym_envs import MAX_STEP, ReachEnv

    env = ReachEnv()
    ok, steps, final = [], [], []
    for s in seeds:
        obs, _ = env.reset(seed=s)
        while True:
            a = np.clip(obs[6:9] / MAX_STEP, -1, 1) if policy is None else policy.act(obs, deterministic=True)[0]
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        ok.append(term), steps.append(env.t), final.append(info["distance"])
    return {"success_rate": float(np.mean(ok)), "median_steps": float(np.median(steps)),
            "median_final_distance_mm": float(np.median(final) * 1000),
            "max_final_distance_mm": float(np.max(final) * 1000)}


# ---------------------------------------------------------------- a 2-second task for the tests
def build_toy(worker_id: int, **_):
    """1-D 'walk to zero'. Learnable in seconds; used by tests/test_rl.py to check PPO end to end."""
    import gymnasium as gym

    class Toy(gym.Env):
        observation_space = gym.spaces.Box(-np.inf, np.inf, (1,), np.float32)
        action_space = gym.spaces.Box(-1.0, 1.0, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self.x, self.t = float(self.np_random.uniform(-1, 1)), 0
            return np.array([self.x], np.float32), {}

        def step(self, a):
            self.x += 0.2 * float(np.clip(a[0], -1, 1))
            self.t += 1
            done = abs(self.x) < 0.05
            return np.array([self.x], np.float32), -abs(self.x) + (1.0 if done else 0.0), done, self.t >= 20, \
                {"is_success": done}

    env = Toy()
    env.reset(seed=worker_id + 1)
    return env, GaussianPolicy(1, 1, hidden=32)
