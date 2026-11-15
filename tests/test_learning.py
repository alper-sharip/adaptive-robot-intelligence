"""Stage 4: observation/action spaces, demonstrations, behaviour cloning, reactive expert."""

import numpy as np
import pytest
import torch

from ari.envs.pick_place import PickPlaceEnv
from ari.learning import bc, demos
from ari.learning.evaluate import CubeShift, make_policy
from ari.learning.spaces import (ACT_DIM, GRIPPER_DOWN, OBS_DIM, action_from_target, make_obs,
                                 target_from_action, tcp_yaw)
from ari.robot.transforms import make_T, rot_z
from ari.skills.reactive_pick_place import ReactiveExpert


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv()


# ---------- spaces ----------
def test_obs_shape_and_content(env):
    env.reset(0)
    o = make_obs(env)
    assert o.shape == (OBS_DIM,) and o.dtype == np.float32
    assert np.allclose(o[:3], env.tcp_pose()[:3, 3], atol=1e-6)
    assert np.allclose(o[13:16], env.cube_pose()[:3, 3] - env.tcp_pose()[:3, 3], atol=1e-6)


def test_cube_yaw_encoding_respects_symmetry(env):
    """A cube turned by 90 degrees is the same cube: the observation must not change."""
    env.reset(0)
    o0 = make_obs(env)
    c = env.cube_qpos
    yaw = env.cube_yaw() + np.pi / 2
    env.data.qpos[c + 3:c + 7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    import mujoco
    mujoco.mj_forward(env.model, env.data)
    assert np.allclose(make_obs(env), o0, atol=1e-5)


@pytest.mark.parametrize("yaw0,yaw1", [(0.0, 0.3), (1.5, -0.7), (3.0, -3.0)])
def test_action_encode_decode_roundtrip(yaw0, yaw1):
    T0 = make_T(rot_z(yaw0) @ GRIPPER_DOWN, np.array([0.5, 0.1, 0.3]))
    T1 = make_T(rot_z(yaw1) @ GRIPPER_DOWN, np.array([0.52, 0.08, 0.25]))
    a = action_from_target(T0, T1, True)
    assert a.shape == (ACT_DIM,) and abs(a[3]) <= np.pi
    T, closed = target_from_action(T0, a)
    assert closed and np.allclose(T, T1, atol=1e-6)
    assert np.isclose(tcp_yaw(T), tcp_yaw(T1), atol=1e-6)


# ---------- demonstrations ----------
def test_replaying_recorded_actions_reproduces_the_episode(env):
    """Regression: recording a 100 Hz expert and replaying at 20 Hz did NOT reproduce it.
    The expert now acts through the same executor as the learner, so replay is exact."""
    obs, act, ok = demos.record_episode(env, 10_003)
    err = env.placement_error()
    assert ok and len(obs) == len(act)

    class Replay:
        def reset(self):
            self.i = 0

        def __call__(self, o):
            a = act[min(self.i, len(act) - 1)].copy() if self.i < len(act) else np.zeros(5, np.float32)
            self.i += 1
            return a

    r = demos.rollout(env, Replay(), 10_003)
    assert r.success and abs(r.placement_error - err) < 1e-6


def test_scripted_expert_survives_the_policy_interface(env):
    """Regression: with staircase (non-interpolated) commands the cube slipped during the lift."""
    for seed in (10_005, 10_006):   # the two seeds that failed before interpolation
        assert demos.record_episode(env, seed)[2]


def test_demo_seeds_do_not_overlap_evaluation():
    assert demos.TRAIN_SEED_START > 99


# ---------- reactive expert ----------
def test_reactive_expert_is_memoryless_and_succeeds(env):
    ex = ReactiveExpert()
    assert not [k for k in vars(ex) if k.startswith("_")]   # no hidden state
    obs, act, ok = demos.record_reactive(env, 10_001, ex)
    assert ok
    env.reset(10_001)
    assert np.array_equal(ex(env), ex(env))                 # same state -> same action


def test_reactive_expert_is_continuous_near_the_place_funnel(env):
    """Regression: with hard thresholds, moving the held cube from 2.9 to 3.1 mm off target
    flipped the expert from 'lower' to 'climb 13 cm'. Small state change -> small action change."""
    import mujoco

    ex = ReactiveExpert()
    r = demos.record_reactive(env, 10_001, ex, max_time=4.0)   # stop while carrying
    assert r is not None
    base = env.target_pos().copy()

    def action_with_target_shift(dx):
        env.model.site_pos[env.target_site, 0] = base[0] + dx
        mujoco.mj_forward(env.model, env.data)
        return ex(env)

    a0, a1 = action_with_target_shift(0.0), action_with_target_shift(0.0005)
    env.model.site_pos[env.target_site, 0] = base[0]
    assert np.linalg.norm(a1[:3] - a0[:3]) < 0.004


def test_reactive_expert_recovers_from_moved_cube_but_scripted_does_not(env):
    seed = 3
    r = demos.rollout(env, make_policy("reactive", env), seed, disturbance=CubeShift(seed))
    assert r.success
    r = demos.rollout(env, make_policy("scripted", env), seed, disturbance=CubeShift(seed), max_time=14)
    assert not r.success


# ---------- behaviour cloning ----------
def _toy_data(n_eps=40, T=50, seed=0):
    rng = np.random.default_rng(seed)
    obs = rng.normal(size=(n_eps * T, OBS_DIM)).astype(np.float32)
    act = np.zeros((n_eps * T, ACT_DIM), np.float32)
    act[:, :4] = 0.01 * obs[:, :4] + 0.002          # a learnable linear rule
    act[:, 4] = (obs[:, 5] > 0).astype(np.float32)   # gripper depends on one feature
    return {"observation.state": obs, "action": act, "episode_index": np.repeat(np.arange(n_eps), T)}


def test_split_is_by_episode():
    ep = np.repeat(np.arange(20), 40)
    tr, va = bc.split_by_episode(ep, 0.1, 0)
    assert not set(ep[tr]) & set(ep[va]) and va.sum() == 2 * 40


def test_bc_learns_a_simple_rule_and_is_reproducible():
    data = _toy_data()
    tc = dict(epochs=40, hidden=64, layers=2, batch_size=32)
    pol, cfg = bc.train(data, bc.TrainConfig(**tc), verbose=False)
    assert cfg.history[-1]["val_motion"] < 0.1   # normalized MSE; predicting the mean gives 1.0
    o = data["observation.state"][0]
    a = pol(o)
    assert np.allclose(a[:4], data["action"][0, :4], atol=0.004) and a[4] == data["action"][0, 4]
    pol2, _ = bc.train(data, bc.TrainConfig(**tc), verbose=False)
    assert np.allclose(pol2(o), a)


def test_policy_save_load_roundtrip(tmp_path):
    pol, _ = bc.train(_toy_data(), bc.TrainConfig(epochs=2, hidden=32, layers=2), verbose=False)
    pol.save(tmp_path / "p.pt")
    o = torch.randn(OBS_DIM).numpy()
    assert np.allclose(bc.BCPolicy.load(tmp_path / "p.pt")(o), pol(o))


# ---------- DAgger data hygiene ----------
def test_drop_repeats_removes_stuck_states_only():
    """Regression: a stuck learner records one state hundreds of times; those copies made
    DAgger from a weak policy worse every iteration (71% -> 25%)."""
    from ari.learning.evaluate import drop_repeats

    moving = np.cumsum(np.full((50, OBS_DIM), 0.01, np.float32), axis=0)   # changes every step
    stuck = np.repeat(moving[-1:], 200, axis=0)                           # then frozen
    obs = np.concatenate([moving, stuck])
    act = np.arange(len(obs), dtype=np.float32)[:, None].repeat(ACT_DIM, 1)
    o, a = drop_repeats(obs, act)
    assert len(o) == 50 and np.array_equal(a[:, 0], np.arange(50))


# ---------- the trained policy shipped in the repo ----------
def test_shipped_policy_picks_and_places_and_recovers(env):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "outputs" / "models" / "bc_dagger.pt"
    if not path.exists():
        pytest.skip("trained model not present")
    pol = bc.BCPolicy.load(path)
    assert demos.rollout(env, pol, 3).success
    assert demos.rollout(env, pol, 3, disturbance=CubeShift(3)).success
