"""The camera loop: cube and target from images, every policy step (perception/tracking.py)."""

import numpy as np
import pytest

from ari import benchmark as B
from ari.learning import demos
from ari.learning.evaluate import make_policy


@pytest.fixture(scope="module")
def env():
    e = B.BenchEnv(B.get_condition("cam:nominal"))
    yield e
    e.tracker.close()


def _truth(env):
    return env.data.xpos[env.cube_body].copy()


def test_camera_condition_names_resolve():
    c = B.get_condition("cam:moved")
    assert c.camera and c.shift_cube and c.label.endswith("camera")
    assert not B.get_condition("moved").camera


def test_at_reset_cube_and_target_are_located_from_images(env):
    env.reset(3)
    assert env.tracker.stats.target_found
    assert np.linalg.norm(env.target_pos()[:2] - env.data.site_xpos[env.target_site][:2]) < 0.003
    assert np.linalg.norm(env.cube_pose()[:3, 3] - _truth(env)) < 0.003


def test_success_is_judged_on_the_truth_not_on_what_the_policy_is_told(env):
    env.reset(3)
    env.tracker.target = env.tracker.target + np.array([0.05, 0.0, 0.0])     # the camera is now 5 cm off
    true_err = np.linalg.norm(_truth(env)[:2] - env.data.site_xpos[env.target_site][:2])
    assert env.placement_error() == pytest.approx(true_err)


def test_the_cube_is_tracked_while_it_is_carried(env):
    """Run the reactive policy on camera input until it carries the cube, then compare."""
    env.reset(1)
    pol = make_policy("reactive", env)
    ex = demos.ActionExecutor(env, joint_interp=True)
    carried = []
    for _ in range(160):
        ref = env.tcp_pose().copy()
        ex.step(pol(demos.make_obs(env)), ref)
        if env.tracker._holding() and _truth(env)[2] > 0.05:
            carried.append(np.linalg.norm(env.cube_pose()[:3, 3] - _truth(env)))
    assert len(carried) > 10
    assert np.percentile(carried, 95) < 0.003


def test_when_no_camera_sees_the_cube_it_moves_with_the_hand(env):
    env.reset(1)
    pol = make_policy("reactive", env)
    ex = demos.ActionExecutor(env, joint_interp=True)
    for _ in range(160):                             # until it is lifting the cube
        ex.step(pol(demos.make_obs(env)), env.tcp_pose().copy())
        if env.tracker._holding() and _truth(env)[2] > 0.06:
            break
    wrist = env.tracker.cams.pop("wrist")            # both cameras blind from here on
    env.tracker.use_front = False
    try:
        for _ in range(10):
            ex.step(np.array([0.01, 0.01, 0.0, 0.0, 1.0], dtype=np.float32), env.tcp_pose().copy())
            assert np.linalg.norm(env.cube_pose()[:3, 3] - _truth(env)) < 0.005
        assert env.tracker.stats.source["held"] >= 10
    finally:
        env.tracker.cams["wrist"] = wrist
        env.tracker.use_front = True


def test_a_learned_policy_succeeds_on_camera_input(env):
    pol = make_policy("outputs/models/bc_reactive_500.pt", env)
    assert sum(demos.rollout(env, pol, s, max_time=20, joint_interp=True).success for s in range(3)) == 3


def test_a_cube_only_partly_in_the_wrist_view_is_not_trusted():
    """Seed 14 of the outside-area condition: the wrist camera sees part of the cube (its
    estimate was 20 mm off). The tracker must fall back to the front camera."""
    from ari.perception.detect import localize_cube
    from ari.perception.tracking import fully_in_view

    env = B.BenchEnv(B.get_condition("cam:outside"))
    try:
        env.reset(14)
        rgb, depth, cam, _ = env.tracker.cams["wrist"].capture(env.data)
        e = localize_cube(rgb, depth, cam, env.cfg.cube_half_size)
        assert e.ok and not fully_in_view(e.position, env.cfg.cube_half_size, cam)
        assert env.tracker.stats.source["front"] == 1
        assert np.linalg.norm(env.cube_pose()[:3, 3] - env.data.xpos[env.cube_body]) < 0.002
    finally:
        env.tracker.close()
