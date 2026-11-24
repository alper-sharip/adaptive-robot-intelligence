"""The interactive slider tool: its math must be exact, and the controls must do what they say."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from ari.interactive import SliderSession, elbow_from_distance, shoulder_wrist_distance, swivel_angle
from ari.robot.ik import JOINT_LIMITS, solve_ik
from ari.robot.kinematics import tcp_pose


def test_closed_form_elbow_angle_is_exact():
    """One root of q4 = delta +/- arccos((A - d^2)/2R) is the true joint-4 angle, for any posture."""
    rng = np.random.default_rng(0)
    for _ in range(3000):
        q = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        e = elbow_from_distance(shoulder_wrist_distance(q))
        assert min(abs(r - q[3]) for r in e["roots_in_range"]) < 1e-9


def test_elbow_ambiguity_is_where_the_geometry_says():
    e = elbow_from_distance(shoulder_wrist_distance(np.array([0, 0, 0, -0.2, 0, 1.5, 0])))
    assert e["ambiguous"] and np.degrees(e["q4_straight"]) == pytest.approx(-26.8, abs=0.1)
    assert not elbow_from_distance(shoulder_wrist_distance(np.array([0, 0, 0, -2.0, 0, 1.5, 0])))["ambiguous"]


def test_ik_trace_records_every_iteration():
    trace = []
    T = tcp_pose(np.array([0.3, 0.2, -0.1, -2.0, 0.1, 2.0, 0.5]))
    res = solve_ik(T, trace=trace)
    assert res.success and len(trace) == res.iterations and trace[-1][0] < 1e-4


@pytest.fixture(scope="module")
def session():
    return SliderSession(3)


def test_cartesian_sliders_reach_the_target(session):
    session.reset(3)
    session.set_cartesian(x=0.55, y=0.10, z=0.20, yaw=0.4)
    session.step(150)
    st = session.state()
    assert st["ik"]["success"] and st["tcp_error_to_target_mm"] < 0.5


def test_swivel_moves_the_elbow_but_not_the_gripper(session):
    session.reset(3)
    session.set_cartesian(x=0.5, y=0.1, z=0.25, yaw=0.3)
    session.step(150)
    T0 = np.array(session.state()["T_mujoco"])
    for target in (40.0, -40.0):
        session.set_cartesian(swivel=target)
        session.step(250)
        st = session.state()
        assert st["swivel_deg"] == pytest.approx(target, abs=1.0)
        assert np.linalg.norm(np.array(st["T_mujoco"])[:3, 3] - T0[:3, 3]) < 0.0005


def test_state_shows_live_angles_and_agrees_with_mujoco(session):
    session.reset(3)
    session.set_joints(q=np.radians([20, 30, -10, -100, 15, 120, 30]))
    session.step(120)
    st = json.loads(json.dumps(session.state()))          # what the page receives
    q = np.array(st["q"])
    assert [r["theta"] for r in st["dh"]] == pytest.approx(list(q))
    assert st["fk_vs_mujoco_um"] < 0.01
    assert np.allclose(np.array(st["frames"][-1]) @ np.array(st["flange"]) @ np.array(st["tcp_offset"]), st["T_fk"])
    assert st["swivel_deg"] == pytest.approx(np.degrees(swivel_angle(q)))


def test_server_applies_commands_and_renders_a_frame(session):
    spec = importlib.util.spec_from_file_location("slider_control", Path(__file__).parents[1] / "scripts" / "slider_control.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    session.reset(3)
    mod.apply(session, {"type": "cartesian", "x": 520, "y": 50, "z": 200, "yaw": 30, "gripper": 0})
    assert session.target["x"] == pytest.approx(0.52) and session.gripper == 0
    import mujoco

    r = mujoco.Renderer(session.env.model, height=120, width=160)
    try:
        jpg = mod.render(session, r, mod.Shared().camera, session.state())
    finally:
        r.close()
    assert jpg[:2] == b"\xff\xd8" and len(jpg) > 1000
