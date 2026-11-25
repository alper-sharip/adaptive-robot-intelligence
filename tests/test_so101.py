"""SO-101 (simulated): own kinematics against MuJoCo, closed-form IK, and scripted pick-place."""

import ari  # noqa: F401  (sets up headless rendering before mujoco is imported)

import mujoco
import numpy as np
import pytest

from ari.so101.kinematics import JOINTS, fk, ik_top_down, tool_yaw

try:
    from ari.so101.env import SO101Env
    ENV = SO101Env()
except Exception as err:  # Menagerie not available
    pytest.skip(f"SO-101 model not available: {err}", allow_module_level=True)

from ari.so101.pick_place import ScriptedPickPlace  # noqa: E402


def _mujoco_tool(env, q):
    env.data.qpos[env.arm_qpos] = q
    mujoco.mj_kinematics(env.model, env.data)
    sid = env.tool_site
    T = np.eye(4)
    T[:3, :3] = env.data.site_xmat[sid].reshape(3, 3)
    T[:3, 3] = env.data.site_xpos[sid]
    return T


def _random_q(env, rng):
    lo, hi = env.model.jnt_range[[env.model.joint(j).id for j in JOINTS]].T
    return rng.uniform(lo, hi)


def test_fk_matches_mujoco():
    rng = np.random.default_rng(0)
    for _ in range(50):
        q = _random_q(ENV, rng)
        assert np.abs(fk(ENV.chain, q) - _mujoco_tool(ENV, q)).max() < 1e-9


def test_geometry_measured():
    g = ENV.geometry
    assert g.L1 == pytest.approx(0.116, abs=1e-3)
    assert g.L2 == pytest.approx(0.135, abs=1e-3)


@pytest.mark.parametrize("p,yaw", [((0.20, 0.0, 0.0125), 0.0), ((0.15, 0.10, 0.03), 0.7),
                                   ((0.22, -0.12, 0.06), -1.2), ((0.12, 0.05, 0.0125), 1.5)])
def test_ik_top_down_exact(p, yaw):
    sol = ik_top_down(ENV.chain, ENV.geometry, np.array(p), yaw)
    assert sol.ok, sol.reason
    T = fk(ENV.chain, sol.q)
    assert np.linalg.norm(T[:3, 3] - p) < 1e-9
    assert T[2, 0] == pytest.approx(-1.0, abs=1e-9)      # tool x axis points straight down
    d = (tool_yaw(T) - yaw + np.pi) % (2 * np.pi) - np.pi
    assert abs(d) < 1e-9


def test_ik_respects_limits_and_rejects_far_targets():
    lo, hi = ENV.model.jnt_range[[ENV.model.joint(j).id for j in JOINTS]].T
    sol = ik_top_down(ENV.chain, ENV.geometry, np.array([0.18, 0.05, 0.02]), 0.3)
    assert sol.ok and np.all(sol.q >= lo - 1e-9) and np.all(sol.q <= hi + 1e-9)
    assert not ik_top_down(ENV.chain, ENV.geometry, np.array([0.45, 0.0, 0.0125]), 0.0).ok


def test_scripted_pick_place_and_calibration():
    cal = ScriptedPickPlace.calibrate(ENV)
    assert 0.007 < np.linalg.norm(cal.held) < 0.012            # one-sided gripper pushes the cube ~1 cm
    plain, comp = ScriptedPickPlace(ENV), ScriptedPickPlace(ENV, calibration=cal)
    e_plain, e_comp = [], []
    for seed in range(4):
        r = plain.run(seed)
        assert r.success, r.failure
        e_plain.append(r.placement_error_mm)
        r = comp.run(seed)
        assert r.success, r.failure
        e_comp.append(r.placement_error_mm)
    assert np.median(e_comp) < np.median(e_plain)
