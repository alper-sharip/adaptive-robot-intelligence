"""Stage 5: Gymnasium environments, PPO pieces, residual policy."""

import numpy as np
import pytest
import torch

from ari.envs.gym_envs import ACTION_SCALE, PickPlaceGymEnv, ReachEnv, to_normalized, to_physical
from ari.rl import ppo, tasks


@pytest.fixture(scope="module")
def pick_env():
    return PickPlaceGymEnv()


@pytest.fixture(scope="module")
def reach_env():
    return ReachEnv()


# ---------- environments ----------
@pytest.mark.parametrize("fixture", ["pick_env", "reach_env"])
def test_environments_follow_the_gymnasium_api(fixture, request):
    from gymnasium.utils.env_checker import check_env

    check_env(request.getfixturevalue(fixture), skip_render_check=True)


def test_reset_is_deterministic_per_seed(pick_env):
    a, _ = pick_env.reset(seed=5)
    b, _ = pick_env.reset(seed=5)
    c, _ = pick_env.reset(seed=6)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_time_limit_is_truncation_not_termination(reach_env):
    reach_env.reset(seed=1)
    for _ in range(reach_env.max_steps):
        _, _, term, trunc, _ = reach_env.step(np.zeros(3, np.float32))
    assert trunc and not term


def test_shaping_telescopes(pick_env):
    """Potential-based shaping: with gamma = 1 the shaping rewards of ANY trajectory sum to
    Phi(end) - Phi(start), so the agent cannot collect reward by wandering in circles."""
    env = PickPlaceGymEnv(gamma=1.0)
    env.reset(seed=3)
    phi0 = env.phi
    rng = np.random.default_rng(0)
    total = 0.0
    for _ in range(40):
        _, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 5).astype(np.float32))
        total += r + 0.01                       # remove the per-step cost
    assert not term and np.isclose(total, env.phi - phi0, atol=1e-6)


def test_reach_p_controller_solves_reach(reach_env):
    r = tasks.evaluate_reach(None, seeds=range(5))
    assert r["success_rate"] == 1.0


def test_action_scaling_roundtrip():
    phys = np.array([0.01, -0.02, 0.005, 0.1, 1.0], np.float32)
    assert np.allclose(to_physical(to_normalized(phys)), phys, atol=1e-7)
    assert np.allclose(to_physical(np.array([5, -5, 0, 0, -3.0]))[:3], [ACTION_SCALE[0], -ACTION_SCALE[1], 0])


# ---------- PPO pieces ----------
def test_gae_matches_hand_calculation():
    # two steps, then the episode ends; gamma = 0.5, lam = 1 -> advantages are discounted returns minus values
    adv, ret = ppo.gae(rewards=[1.0, 2.0], values=[0.5, 1.0], dones=[False, True], last_value=99.0, gamma=0.5, lam=1.0)
    assert np.allclose(ret, [1.0 + 0.5 * 2.0, 2.0]) and np.allclose(adv, [2.0 - 0.5, 2.0 - 1.0])
    # no episode end: the value of the state after the last step is bootstrapped
    adv, ret = ppo.gae([1.0], [0.0], [False], last_value=4.0, gamma=0.5, lam=0.9)
    assert np.allclose(ret, [1.0 + 0.5 * 4.0])


def test_running_normalization_matches_batch_statistics():
    pol = ppo.GaussianPolicy(3, 1)
    x = torch.randn(500, 3) * torch.tensor([1.0, 5.0, 0.1]) + torch.tensor([0.0, 2.0, -1.0])
    pol.update_norm(x[:200])
    pol.update_norm(x[200:])
    assert torch.allclose(pol.obs_mean, x.mean(0), atol=1e-4) and torch.allclose(pol.obs_std, x.std(0), rtol=2e-2)


def test_ppo_learns_a_toy_task():
    cfg = ppo.PPOConfig(total_steps=24_000, steps_per_worker=1000, workers=2, minibatch=100, lr=3e-3, gamma=0.95)
    ppo.train(tasks.build_toy, {}, cfg, verbose=False)
    assert cfg.log[0]["success"] < 0.7 and cfg.log[-1]["success"] > 0.9


# ---------- residual policy ----------
def test_residual_policy_starts_exactly_at_the_cloned_policy(pick_env):
    from pathlib import Path

    from ari.learning.bc import BCPolicy

    path = Path(__file__).resolve().parents[1] / "outputs" / "models" / "bc_dagger.pt"
    if not path.exists():
        pytest.skip("trained model not present")
    res = tasks.ResidualPolicy(torch.load(path, weights_only=True))
    base = BCPolicy.load(path)
    obs, _ = pick_env.reset(seed=2)
    for _ in range(5):
        a_res = to_physical(res.act(obs, deterministic=True)[0])
        b = base(obs)
        b[:4] = np.clip(b[:4], -ACTION_SCALE, ACTION_SCALE)      # the action space clips at one full step
        assert np.allclose(a_res, b, atol=1e-6)
        obs, *_ = pick_env.step(to_normalized(a_res))
    assert all(not p.requires_grad for p in res.base.parameters())      # the clone is frozen


def test_rl_checkpoint_roundtrip(tmp_path):
    pol = ppo.GaussianPolicy(20, 5)
    tasks.save(tmp_path / "p.pt", pol, "ppo_pick")
    obs = np.random.default_rng(0).normal(size=20).astype(np.float32)
    assert np.allclose(tasks.load(tmp_path / "p.pt").act(obs, deterministic=True)[0], pol.act(obs, deterministic=True)[0])
