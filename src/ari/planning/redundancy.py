"""M8: using the Panda's 7th degree of freedom to avoid obstacles.

A 6-D gripper pose leaves a 1-D family of arm postures (the elbow can swing around the
shoulder-wrist line while the hand stays fixed). Plain IK picks whichever posture its
null-space term pulls toward. Near an obstacle that may be the one posture that collides.

`find_free_posture`  : IK from many different starting postures; keep a collision-free one.
`cartesian_path`     : convert a straight-line gripper move into a dense joint path that
                       KEEPS the chosen posture, collision-checking every sample.
                       (The same idea as MoveIt's computeCartesianPath.)
"""

from __future__ import annotations

import numpy as np

from ari.planning.collision import CollisionChecker
from ari.planning.trajectory import PoseTrajectory
from ari.robot.ik import JOINT_LIMITS, Q_HOME, solve_ik


def find_free_posture(T: np.ndarray, cc: CollisionChecker, q_seed: np.ndarray | None = None,
                      n_restarts: int = 40, seed: int = 0, spread: float = 0.8):
    """Return a collision-free IK solution for pose T, or None.

    Tries q_seed first (smallest motion), then random postures around home. Each restart
    uses its own start as the null-space 'preferred' posture, so different restarts land
    on genuinely different elbow positions instead of all converging to the same one.
    """
    rng = np.random.default_rng(seed)
    starts = [] if q_seed is None else [np.asarray(q_seed, float)]
    starts += [np.clip(Q_HOME + rng.normal(0, spread, 7), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
               for _ in range(n_restarts)]
    for q0 in starts:
        res = solve_ik(T, q_init=q0, q_preferred=q0)
        if res.success and cc.is_free(res.q):
            return res.q
    return None


def cartesian_path(T0: np.ndarray, T1: np.ndarray, q_start: np.ndarray, cc: CollisionChecker,
                   n: int = 25, max_joint_jump: float = 0.2):
    """Dense joint path following the straight Cartesian move T0 -> T1 from posture q_start.

    Returns a list of n joint configurations, or None if any sample fails IK, collides,
    or jumps more than `max_joint_jump` rad from the previous one (a sign IK flipped to a
    different solution branch, which would sweep the arm through unchecked space).
    """
    traj = PoseTrajectory(T0, T1, 1.0)
    q_prev = np.asarray(q_start, float)
    path = [q_prev]
    for s in np.linspace(0, 1, n)[1:]:
        T = _pose_at_s(traj, s)  # equally spaced along the line (DenseJointPath adds the timing)
        res = solve_ik(T, q_init=q_prev, q_preferred=q_start)
        if not res.success or np.abs(res.q - q_prev).max() > max_joint_jump or not cc.is_free(res.q):
            return None
        path.append(res.q)
        q_prev = res.q
    return path


def _pose_at_s(traj: PoseTrajectory, s: float) -> np.ndarray:
    """Pose at path parameter s in [0, 1] (linear in s, unlike traj(t) which is min-jerk in t)."""
    from ari.robot.transforms import axis_angle_to_rot, make_T

    p = (1 - s) * traj.T0[:3, 3] + s * traj.T1[:3, 3]
    R = traj.T0[:3, :3] @ axis_angle_to_rot(s * traj._dR)
    return make_T(R, p)


def anchored_paths(T_anchor: np.ndarray, T_others: list, cc: CollisionChecker,
                   q_seed: np.ndarray | None = None, n_restarts: int = 40, seed: int = 0,
                   spread: float = 0.8):
    """Choose a posture at the MOST CONSTRAINED pose (e.g. the grasp, down by the wall)
    and trace straight-line paths from it to each pose in T_others.

    Why anchor at the bottom? Choosing the posture at the top of a vertical move and holding
    it on the way down failed near the wall: the elbow drifted toward the wall as the hand
    descended. The bottom is where clearance is tightest, so that is where the posture
    choice matters. Returns (q_anchor, [path_to_each_other]) or None.
    """
    rng = np.random.default_rng(seed)
    starts = [] if q_seed is None else [np.asarray(q_seed, float)]
    starts += [np.clip(Q_HOME + rng.normal(0, spread, 7), JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])
               for _ in range(n_restarts)]
    for q0 in starts:
        res = solve_ik(T_anchor, q_init=q0, q_preferred=q0)
        if not res.success or not cc.is_free(res.q):
            continue
        paths = [cartesian_path(T_anchor, T, res.q, cc) for T in T_others]
        if all(p is not None for p in paths):
            return res.q, paths
    return None
