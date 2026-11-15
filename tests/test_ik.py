"""IK must reach targets we know are reachable (because we generated them with FK)."""

import numpy as np

from ari.robot.ik import JOINT_LIMITS, Q_HOME, solve_ik
from ari.robot.kinematics import tcp_pose
from ari.robot.transforms import make_T, rot_x


def test_ik_random_reachable_targets():
    """Sample random joint configs near a typical working posture, compute where the TCP is,
    then ask IK to find a solution from the home posture. Report the success rate."""
    rng = np.random.default_rng(42)
    n, successes, errors = 100, 0, []
    for _ in range(n):
        q_true = np.clip(Q_HOME + rng.uniform(-0.8, 0.8, 7), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
        T = tcp_pose(q_true)
        res = solve_ik(T, q_init=Q_HOME, max_iters=300)
        successes += res.success
        if res.success:
            errors.append(res.pos_error)
            # The solution must respect joint limits and actually produce the pose.
            assert np.all(res.q >= JOINT_LIMITS[:, 0] - 1e-9)
            assert np.all(res.q <= JOINT_LIMITS[:, 1] + 1e-9)
            assert np.linalg.norm(tcp_pose(res.q)[:3, 3] - T[:3, 3]) < 1e-4
    rate = successes / n
    print(f"\nIK success rate: {rate:.0%}, mean position error {1000 * np.mean(errors):.3f} mm")
    assert rate >= 0.95


def test_ik_top_down_grasp_poses():
    """The poses we'll actually use for pick-and-place: gripper pointing down over a table."""
    down = rot_x(np.pi)
    # x up to 0.65 m: at (0.7, ±0.3, 0.4) a top-down pose is just out of reach
    # (see test_workspace_edge_is_unreachable_top_down).
    for x in np.linspace(0.3, 0.65, 5):
        for y in np.linspace(-0.3, 0.3, 5):
            for z in (0.05, 0.2, 0.4):
                res = solve_ik(make_T(down, np.array([x, y, z])))
                assert res.success, f"failed at {(x, y, z)}: {res}"


def test_workspace_edge_is_unreachable_top_down():
    """Found during development: with the gripper pointing down, (0.7, 0.3, 0.4) is ~8.7 mm
    beyond full extension. 50 random restarts all stop at the same distance, which is
    how we know it's the workspace boundary and not the solver getting stuck."""
    T = make_T(rot_x(np.pi), np.array([0.7, 0.3, 0.4]))
    rng = np.random.default_rng(0)
    best = min(
        (solve_ik(T, q_init=rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1]), max_iters=500)
         for _ in range(20)),
        key=lambda r: r.pos_error,
    )
    assert not best.success
    assert 0.005 < best.pos_error < 0.012


def test_ik_reports_failure_for_unreachable_target():
    res = solve_ik(make_T(np.eye(3), np.array([2.0, 0.0, 0.5])))  # 2 m away, arm reaches ~0.85 m
    assert not res.success
