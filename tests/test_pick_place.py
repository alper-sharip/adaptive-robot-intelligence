"""Pick-and-place environment and scripted baseline.

Several of these are regression tests for bugs found during development
(docs/devlog.md). If one fails after a change, the bug has come back.
"""

import numpy as np
import pytest

from ari.envs.pick_place import PickPlaceEnv, SceneConfig
from ari.skills.pick_place import ScriptedPickPlace, grasp_yaw, run_episode


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv()


def test_reset_is_deterministic_and_seeds_differ(env):
    a, b, c = env.sample_episode(7), env.sample_episode(7), env.sample_episode(8)
    assert np.array_equal(a.cube_xy, b.cube_xy) and a.cube_yaw == b.cube_yaw
    assert not np.array_equal(a.cube_xy, c.cube_xy)


def test_episodes_respect_sampling_rules(env):
    for seed in range(200):
        ep = env.sample_episode(seed)
        assert np.linalg.norm(ep.cube_xy - ep.target_xy) >= env.cfg.min_separation
        (x0, x1), (y0, y1) = env.cfg.cube_region
        assert x0 <= ep.cube_xy[0] <= x1 and y0 <= ep.cube_xy[1] <= y1


def test_reset_places_cube_where_specified(env):
    ep = env.reset(3)
    assert np.allclose(env.cube_pose()[:2, 3], ep.cube_xy)
    assert np.isclose(env.cube_yaw(), (ep.cube_yaw + np.pi) % (2 * np.pi) - np.pi)
    assert np.allclose(env.target_pos()[:2], ep.target_xy)


@pytest.mark.parametrize("yaw", np.linspace(-np.pi, np.pi, 17))
def test_grasp_yaw_is_equivalent_and_small(yaw):
    g = grasp_yaw(yaw)
    assert -np.pi / 4 - 1e-9 <= g <= np.pi / 4 + 1e-9
    k = (yaw - g) / (np.pi / 2)
    assert np.isclose(k, round(k))  # differs from the cube yaw by a multiple of 90 deg


def test_scripted_baseline_succeeds(env):
    for seed in range(3):
        r = run_episode(env, ScriptedPickPlace(), seed)
        assert r.success, r
        assert r.placement_error < 0.005
        assert r.ik_failures == 0


def test_episode_is_reproducible(env):
    r1 = run_episode(env, ScriptedPickPlace(), 5)
    r2 = run_episode(env, ScriptedPickPlace(), 5)
    assert r1.placement_error == r2.placement_error and r1.slip == r2.slip


def test_cube_friction_setting_takes_effect():
    """Regression: MuJoCo uses max(friction) of the two geoms unless priorities differ, so the
    cube's friction was silently ignored. With 1 N per finger, mu=0.3 cannot lift 0.98 N."""
    r = run_episode(PickPlaceEnv(SceneConfig(cube_friction=0.3)), ScriptedPickPlace(), 0)
    assert r.failure == "grasp_failed"


def test_stiff_gripper_keeps_damping_ratio():
    """Regression: scaling gripper kp without kd made the fingers slam into the cube and
    penetrate it at high speed (7% success). kd now scales with sqrt(kp)."""
    env = PickPlaceEnv(SceneConfig(gripper_stiffness_scale=10))
    r = run_episode(env, ScriptedPickPlace(max_speed=1.0), 0)
    assert r.success, r


def test_elliptic_cone_reduces_creep():
    """MuJoCo's default pyramidal friction cone lets a held cube creep ~3 mm during transport."""
    pyr = run_episode(PickPlaceEnv(SceneConfig(elliptic_cone=False, impratio=1.0)), ScriptedPickPlace(), 3)
    ell = run_episode(PickPlaceEnv(SceneConfig()), ScriptedPickPlace(), 3)
    assert ell.slip < pyr.slip / 3
