"""Our from-scratch kinematics must agree with MuJoCo's, which is built from a different
description of the same robot (MJCF body offsets vs Franka's DH table)."""

import mujoco
import numpy as np
import pytest

from ari.robot.kinematics import hand_pose, tcp_jacobian, tcp_pose
from ari.robot.model import TCP_OFFSET, arm_dof_indices, arm_joint_limits, arm_qpos_indices, load_panda
from ari.robot.transforms import quat_to_rot, rot_to_axis_angle, rot_x, rot_z

N_SAMPLES = 200


@pytest.fixture(scope="module")
def panda():
    model, data = load_panda()
    return model, data, arm_qpos_indices(model), arm_dof_indices(model), arm_joint_limits(model)


def random_configs(limits, n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.uniform(limits[:, 0], limits[:, 1], size=(n, 7))


def test_fk_matches_mujoco(panda):
    model, data, qi, _, limits = panda
    hand = model.body("hand").id
    for q in random_configs(limits, N_SAMPLES):
        data.qpos[qi] = q
        mujoco.mj_forward(model, data)
        T = hand_pose(q)
        assert np.allclose(T[:3, 3], data.xpos[hand], atol=1e-9), "position mismatch"
        # MJCF stores quaternions rounded to 8 digits, so allow ~1e-7 in rotation.
        assert np.allclose(T[:3, :3], data.xmat[hand].reshape(3, 3), atol=1e-6), "rotation mismatch"


def test_jacobian_matches_mujoco(panda):
    model, data, qi, vi, limits = panda
    hand = model.body("hand").id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for q in random_configs(limits, 50, seed=1):
        data.qpos[qi] = q
        mujoco.mj_forward(model, data)
        tcp_world = data.xpos[hand] + data.xmat[hand].reshape(3, 3) @ np.array([0, 0, TCP_OFFSET])
        mujoco.mj_jac(model, data, jacp, jacr, tcp_world, hand)
        J_mj = np.vstack([jacp[:, vi], jacr[:, vi]])
        assert np.allclose(tcp_jacobian(q), J_mj, atol=1e-6)


def test_jacobian_matches_finite_differences(panda):
    """Independent of MuJoCo: nudge each joint and see how the TCP moves."""
    *_, limits = panda
    eps = 1e-6
    for q in random_configs(limits, 20, seed=2):
        J = tcp_jacobian(q)
        T0 = tcp_pose(q)
        for i in range(7):
            dq = np.zeros(7)
            dq[i] = eps
            T1 = tcp_pose(q + dq)
            v = (T1[:3, 3] - T0[:3, 3]) / eps
            w = rot_to_axis_angle(T1[:3, :3] @ T0[:3, :3].T) / eps
            assert np.allclose(J[:3, i], v, atol=1e-5)
            assert np.allclose(J[3:, i], w, atol=1e-5)


@pytest.mark.parametrize("angle", [0.0, 1e-9, 0.3, 1.5, 3.0, np.pi - 1e-7, np.pi])
def test_axis_angle_roundtrip(angle):
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    # Build R from axis-angle with Rodrigues' formula, then recover it.
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
    w = rot_to_axis_angle(R)
    assert np.isclose(np.linalg.norm(w), angle, atol=1e-6)
    if angle > 1e-6:
        # At exactly pi, axis and -axis describe the same rotation.
        assert np.allclose(w / np.linalg.norm(w), axis, atol=1e-5) or (
            np.isclose(angle, np.pi, atol=1e-6) and np.allclose(w / np.linalg.norm(w), -axis, atol=1e-5)
        )


def test_quat_to_rot():
    # 90 deg about z
    q = np.array([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])
    assert np.allclose(quat_to_rot(q), rot_z(np.pi / 2))
    q = np.array([0.0, 1.0, 0.0, 0.0])  # 180 deg about x
    assert np.allclose(quat_to_rot(q), rot_x(np.pi))
