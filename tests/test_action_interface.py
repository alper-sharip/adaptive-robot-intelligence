"""The latency fix: an action's step is relative to the pose the policy observed (Stage 7)."""

import numpy as np
import pytest

from ari.envs.pick_place import PickPlaceEnv
from ari.learning.demos import ActionExecutor, rollout
from ari.learning.evaluate import make_policy


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv()


def _run(env, interface, latency, actions, seed=1):
    env.reset(seed)
    ex = ActionExecutor(env, joint_interp=True, latency_steps=latency, interface=interface)
    poses = []
    for a in actions:
        ref = env.tcp_pose().copy()
        ex.step(a, ref)
        poses.append(env.tcp_pose().copy())
    return poses, ex


def test_without_latency_both_interfaces_give_identical_motion(env):
    acts = [np.array([0.01, -0.005, -0.01, 0.05, 0.0], dtype=np.float32)] * 6
    a, _ = _run(env, "observed", 0, acts)
    b, _ = _run(env, "applied", 0, acts)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


def test_a_late_action_keeps_the_target_it_was_computed_for(env):
    """Constant +1 cm steps in x, 2 policy steps late. With the fixed interface the target
    applied is the pose observed 2 steps earlier + 1 cm. With the old one it is added to the
    pose at application, which has moved on meanwhile."""
    step = np.array([0.01, 0, 0, 0, 0], dtype=np.float32)
    for interface in ("observed", "applied"):
        env.reset(1)
        ex = ActionExecutor(env, joint_interp=True, latency_steps=2, interface=interface)
        refs, applied_at = [], []
        for _ in range(8):
            refs.append(env.tcp_pose().copy())
            applied_at.append(env.tcp_pose().copy())     # the pose when this step's command is applied
            ex.step(step, refs[-1])
        expect_fixed = refs[-3][:3, 3] + [0.01, 0, 0]      # computed 2 steps before the last one
        expect_old = applied_at[-1][:3, 3] + [0.01, 0, 0]
        got = ex.prev[:3, 3]
        if interface == "observed":
            assert np.allclose(got, expect_fixed, atol=1e-6)
        else:
            assert np.allclose(got, expect_old, atol=1e-6)
            assert np.linalg.norm(got - expect_fixed) > 0.005     # it has run ahead of the policy


def test_latency_holds_the_arm_until_the_first_action_arrives(env):
    env.reset(1)
    start = env.tcp_pose().copy()
    big = np.array([0.03, 0, 0, 0, 0], dtype=np.float32)
    _run(env, "observed", 3, [big] * 3)                 # three steps: nothing has arrived yet
    assert np.linalg.norm(env.tcp_pose()[:3, 3] - start[:3, 3]) < 0.002


def test_unknown_interface_is_refused(env):
    with pytest.raises(ValueError):
        ActionExecutor(env, interface="absolute")


def test_the_fix_restores_a_learned_policy_under_100ms_latency(env):
    """The headline check: same policy, same seeds, 100 ms latency, old vs fixed interface."""
    pol = make_policy("outputs/models/bc_reactive_500.pt", env)
    seeds = range(8)
    old = sum(rollout(env, pol, s, max_time=20, joint_interp=True, latency_steps=2, interface="applied").success
              for s in seeds)
    new = sum(rollout(env, pol, s, max_time=20, joint_interp=True, latency_steps=2).success for s in seeds)
    assert new == 8 and new > old


def test_reactive_policy_is_told_the_cube_size():
    from ari.envs.pick_place import SceneConfig

    big = PickPlaceEnv(SceneConfig(cube_half_size=0.025))
    r = rollout(big, make_policy("reactive", big), 0, max_time=20, joint_interp=True)
    assert r.success
