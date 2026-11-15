"""Minimum-jerk trajectories must have the properties we rely on."""

import numpy as np
import pytest

from ari.planning.trajectory import PoseTrajectory, duration_for, min_jerk, min_jerk_velocity
from ari.robot.transforms import axis_angle_to_rot, make_T, rot_to_axis_angle, rot_x, rot_z


def test_min_jerk_boundary_conditions():
    tau = np.linspace(0, 1, 10001)
    s = min_jerk(tau)
    ds = np.gradient(s, tau)
    dds = np.gradient(ds, tau)
    assert s[0] == 0 and np.isclose(s[-1], 1)
    assert abs(ds[0]) < 1e-3 and abs(ds[-1]) < 1e-3       # starts and ends at rest
    assert abs(dds[1]) < 1e-2 and abs(dds[-2]) < 1e-2     # with zero acceleration
    assert np.all(np.diff(s) >= 0)                        # never moves backwards


def test_min_jerk_peak_velocity_is_1_875_times_average():
    T = 2.0
    v = min_jerk_velocity(np.linspace(0, 1, 10001), T)
    assert np.isclose(v.max(), 1.875 / T, rtol=1e-4)


def test_pose_trajectory_endpoints_and_rotation():
    T0 = make_T(rot_x(np.pi), np.array([0.5, 0.0, 0.4]))
    T1 = make_T(rot_z(0.7) @ rot_x(np.pi), np.array([0.4, 0.2, 0.1]))
    traj = PoseTrajectory(T0, T1, 1.5)
    assert np.allclose(traj(0.0), T0)
    assert np.allclose(traj(1.5), T1)
    assert np.allclose(traj(10.0), T1)  # holds the goal after the end
    for t in np.linspace(0, 1.5, 20):
        R = traj(t)[:3, :3]
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)  # still a valid rotation


def test_duration_respects_speed_limit():
    T0 = make_T(np.eye(3), np.zeros(3))
    T1 = make_T(np.eye(3), np.array([0.3, 0.4, 0.0]))  # 0.5 m away
    vmax = 0.25
    traj = PoseTrajectory(T0, T1, duration_for(T0, T1, max_speed=vmax))
    t = np.linspace(0, traj.duration, 2001)
    p = np.array([traj(ti)[:3, 3] for ti in t])
    speed = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(t)
    assert speed.max() <= vmax * 1.001


@pytest.mark.parametrize("angle", [0.0, 0.1, 1.0, 2.5, np.pi - 1e-4])
def test_exp_log_roundtrip(angle):
    w = angle * np.array([0.2, -0.6, 0.77]) / np.linalg.norm([0.2, -0.6, 0.77])
    assert np.allclose(rot_to_axis_angle(axis_angle_to_rot(w)), w, atol=1e-6)
