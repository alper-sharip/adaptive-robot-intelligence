"""M8: collision checking, RRT-Connect, redundancy resolution, planned pick-and-place.

Several are regression tests for problems found during development (docs/devlog.md).
"""

import numpy as np
import pytest

from ari.envs.pick_place import PickPlaceEnv, SceneConfig
from ari.planning.collision import CollisionChecker
from ari.planning.redundancy import anchored_paths
from ari.planning.rrt import path_length, rrt_connect, shortcut
from ari.planning.trajectory import DenseJointPath, JointPathTrajectory
from ari.robot.ik import Q_HOME, solve_ik
from ari.robot.transforms import make_T, rot_x, rot_z
from ari.skills.pick_place import GRIPPER_DOWN, ScriptedPickPlace, grasp_yaw, run_episode
from ari.skills.planned_pick_place import PlannedPickPlace

DOWN = rot_x(np.pi)


@pytest.fixture(scope="module")
def wall_env():
    return PickPlaceEnv(SceneConfig.with_wall())


@pytest.fixture
def cc(wall_env):
    wall_env.reset(0)
    c = CollisionChecker(wall_env.model)
    c.sync(wall_env.data)
    return c


def ik(p, R=DOWN, q_init=Q_HOME):
    return solve_ik(make_T(R, np.array(p)), q_init=q_init).q


# ---------- collision checker ----------
def test_home_is_free_despite_close_links(cc):
    """Regression: a margin applied to self-collision flagged home (links 5 and 7 are 2.3 cm apart)."""
    assert cc.is_free(Q_HOME)


def test_pose_inside_wall_is_flagged(cc):
    assert not cc.is_free(ik([0.5, 0.05, 0.15]))


def test_margin_is_exact(cc, wall_env):
    """Fingertips sit 8.9 mm below the TCP (measured). With the finger axis ALONG the wall
    (so both fingers are over it; across the wall, the open gripper straddles it), the
    clearance is TCP height - 8.9 mm. The 2 cm margin must flip exactly there."""
    h = wall_env.episode.wall_height
    along = rot_z(np.pi / 2) @ DOWN
    assert not cc.is_free(ik([0.5, 0.05, h + 0.025], R=along))  # 16 mm clearance: too close
    assert cc.is_free(ik([0.5, 0.05, h + 0.035], R=along))      # 26 mm clearance: OK


def test_floor_counts_only_real_penetration(cc):
    """Regression: inflated margins were inconsistent across shape types near the floor."""
    assert cc.is_free(ik([0.5, -0.2, 0.015]))      # fingertips ~7 mm above the table: allowed
    assert not cc.is_free(ik([0.5, -0.2, -0.005]))  # fingertips in the table: collision


def test_attached_cube_moves_with_hand(wall_env):
    """Lift the cube in simulation, attach, then move the arm in the checker: the cube follows."""
    env = wall_env
    env.reset(3)
    cc = CollisionChecker(env.model)
    cc.attach("cube", env.data)
    q = ik([0.5, -0.1, 0.3])
    cc._set(q)
    cube = cc.data.xpos[env.cube_body]
    hand = cc.data.xpos[env.hand_body]
    rel0 = cc.T_hand_obj[:3, 3]
    assert np.allclose(cube - hand, cc.data.xmat[env.hand_body].reshape(3, 3) @ rel0, atol=1e-9)


# ---------- RRT-Connect ----------
def test_rrt_straight_line_when_possible(cc):
    q0, q1 = ik([0.5, -0.2, 0.3]), ik([0.45, -0.15, 0.35])
    res = rrt_connect(cc, q0, q1)
    assert res.path is not None and len(res.path) == 2 and res.iterations == 0


def test_rrt_finds_collision_free_path_around_wall(cc):
    q0 = ik([0.5, -0.2, 0.15])
    q1 = ik([0.5, 0.25, 0.15], R=rot_z(np.pi / 2) @ DOWN, q_init=q0)
    assert cc.is_free(q0) and cc.is_free(q1)
    assert not cc.segment_free(q0, q1)  # the straight line is blocked
    res = rrt_connect(cc, q0, q1, seed=1)
    assert res.path is not None
    assert np.allclose(res.path[0], q0) and np.allclose(res.path[-1], q1)
    for a, b in zip(res.path[:-1], res.path[1:]):
        assert cc.segment_free(a, b)
    short = shortcut(cc, res.path, seed=1)
    assert np.allclose(short[0], q0) and np.allclose(short[-1], q1)
    assert path_length(short) <= path_length(res.path) + 1e-9
    for a, b in zip(short[:-1], short[1:]):
        assert cc.segment_free(a, b)


def test_rrt_is_deterministic(cc):
    q0 = ik([0.5, -0.2, 0.15])
    q1 = ik([0.5, 0.25, 0.15], R=rot_z(np.pi / 2) @ DOWN, q_init=q0)
    a, b = rrt_connect(cc, q0, q1, seed=4), rrt_connect(cc, q0, q1, seed=4)
    assert len(a.path) == len(b.path) and all(np.array_equal(x, y) for x, y in zip(a.path, b.path))


# ---------- redundancy ----------
def test_posture_search_avoids_elbow_collision(wall_env):
    """Regression: seed 1's cube is 10.6 cm from the wall. Default IK swings link 5 into the
    wall when reaching down; searching postures anchored at the grasp finds a clear one."""
    env = wall_env
    env.reset(1)
    cc = CollisionChecker(env.model)
    cc.sync(env.data)
    cube = env.cube_pose()[:3, 3]
    # Default IK collides at the grasp for both distinct grasp yaws...
    for off in (0, np.pi / 2):
        R = rot_z(grasp_yaw(env.cube_yaw()) + off) @ GRIPPER_DOWN
        assert not cc.is_free(ik(cube, R=R))
    # ...but a posture search anchored at the grasp finds a clear one.
    found = None
    for k, off in enumerate((0, np.pi / 2, -np.pi / 2, np.pi)):
        R = rot_z(grasp_yaw(env.cube_yaw()) + off) @ GRIPPER_DOWN
        T = make_T(R, cube)
        found = anchored_paths(T, [make_T(R, cube + [0, 0, 0.12])], cc, seed=k)
        if found:
            break
    assert found is not None
    q, (path,) = found
    assert all(cc.is_free(p) for p in path)


# ---------- trajectories ----------
def test_joint_trajectories_hit_their_waypoints():
    a, b, c = Q_HOME, Q_HOME + 0.3, Q_HOME - 0.2
    jt = JointPathTrajectory([a, b, c], max_joint_speed=0.8)
    assert np.allclose(jt(0), a) and np.allclose(jt(jt.seg_starts[1]), b) and np.allclose(jt(jt.duration), c)
    dj = DenseJointPath([a, b, c], 2.0)
    assert np.allclose(dj(0), a) and np.allclose(dj(2.0), c) and np.allclose(dj(1.0), b)


# ---------- planned pick-and-place ----------
@pytest.mark.parametrize("seed", [1, 3])
def test_planned_policy_succeeds_without_touching_wall(wall_env, seed):
    r = run_episode(wall_env, PlannedPickPlace(plan_seed=seed), seed)
    assert r.success, r.failure
    assert r.wall_contact_steps == 0


def test_naive_policy_hits_wall(wall_env):
    r = run_episode(wall_env, ScriptedPickPlace(), 3)
    assert r.wall_contact_steps > 0


def test_planned_policy_works_without_obstacles():
    env = PickPlaceEnv()
    r = run_episode(env, PlannedPickPlace(), 0)
    assert r.success
