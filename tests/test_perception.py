"""Stage 3: camera model, detection and localization, vision-driven policies."""

import mujoco
import numpy as np
import pytest

from ari.envs.pick_place import PickPlaceEnv, SceneConfig
from ari.perception.camera import RGBDCamera
from ari.perception.detect import RED, color_mask, fold_yaw, localize_cube, localize_target
from ari.perception.estimator import VisionPolicy
from ari.perception.noise import NOISE_LEVELS, NoiseModel
from ari.skills.pick_place import ScriptedPickPlace, run_episode

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


@pytest.fixture(scope="module")
def setup():
    env = PickPlaceEnv()
    return env, RGBDCamera(env.model)


def capture(env, cam, seed, gt=False):
    env.reset(seed)
    return cam.capture(env.data, with_gt_segmentation=gt)


# ---------- M10 camera model ----------
def test_camera_looks_at_its_target(setup):
    env, cam = setup
    _, _, cm, _ = capture(env, cam, 0)
    uv, z = cm.project(np.array(env.cfg.camera_target))
    assert np.allclose(uv[0], [cm.width / 2, cm.height / 2], atol=1e-6) and z[0] > 0


def test_project_backproject_roundtrip(setup):
    env, cam = setup
    _, _, cm, _ = capture(env, cam, 0)
    pts = np.random.default_rng(0).uniform([0.3, -0.3, 0.0], [0.7, 0.3, 0.3], (50, 3))
    uv, z = cm.project(pts)
    assert np.allclose(cm.backproject(uv[:, 0], uv[:, 1], z), pts, atol=1e-9)


def test_rendered_depth_matches_geometry(setup):
    """Floor pixels must back-project to z = 0 and cube pixels to at most 4 cm. This checks
    intrinsics, extrinsics, axis conventions AND the pixel-centre convention against the renderer."""
    env, cam = setup
    m = env.model
    _, depth, cm, seg = capture(env, cam, 1, gt=True)
    geom = seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)
    r, c = np.nonzero(geom & (seg[..., 0] == m.geom("floor").id))
    assert np.abs(cm.backproject_pixels(r, c, depth)[:, 2]).max() < 1e-4
    r, c = np.nonzero(geom & (seg[..., 0] == m.geom("cube_geom").id))
    assert np.isclose(cm.backproject_pixels(r, c, depth)[:, 2].max(), 0.04, atol=1e-4)


# ---------- M11 detection & localization ----------
@pytest.mark.parametrize("seed", range(5))
def test_localization_accuracy_without_noise(setup, seed):
    env, cam = setup
    rgb, depth, cm, _ = capture(env, cam, seed)
    c, t = localize_cube(rgb, depth, cm, 0.02), localize_target(rgb, depth, cm)
    assert c.ok and t.ok
    assert np.linalg.norm(c.position - env.cube_pose()[:3, 3]) < 0.0015
    assert abs(fold_yaw(c.yaw - env.cube_yaw())) < np.deg2rad(1.0)
    assert np.linalg.norm(t.position[:2] - env.target_pos()[:2]) < 0.003


def test_floor_reflection_is_rejected_by_depth(setup):
    """The floor is a mirror: some red pixels belong to the cube's reflection. They must exist
    (or this test proves nothing) and must not pull the estimate down."""
    env, cam = setup
    rgb, depth, cm, _ = capture(env, cam, 0)
    r, c = np.nonzero(color_mask(rgb, RED))
    z = cm.backproject_pixels(r, c, depth)[:, 2]
    assert (z < 0.005).sum() > 0
    est = localize_cube(rgb, depth, cm, 0.02)
    assert abs(est.position[2] - 0.02) < 0.001


def test_top_face_method_removes_noise_bias(setup):
    """Regression: choosing 'the highest points' as the top face is biased upward under depth
    noise (+6.8 mm measured). The mean-shift plane estimate is not."""
    env, cam = setup
    bias = {"percentile": [], "plane": []}
    for s in range(10):
        rgb0, d0, cm, _ = capture(env, cam, s)
        rgb, depth = NOISE_LEVELS["poor"].apply(rgb0, d0, np.random.default_rng(s))
        for method in bias:
            est = localize_cube(rgb, depth, cm, 0.02, depth_filter=0, top_method=method)
            bias[method].append(est.position[2] - env.cube_pose()[2, 3])
    assert np.mean(bias["percentile"]) > 0.004
    assert abs(np.mean(bias["plane"])) < 0.002


def test_cube_out_of_view_is_reported(setup):
    env, cam = setup
    env.reset(0)
    adr = env.cube_qpos
    env.data.qpos[adr:adr + 3] = [5.0, 5.0, 0.02]  # far outside the camera's view
    mujoco.mj_forward(env.model, env.data)
    rgb, depth, cm, _ = cam.capture(env.data)
    est = localize_cube(rgb, depth, cm, 0.02)
    assert not est.ok and est.reason == "not_seen"


@pytest.mark.parametrize("yaw", np.linspace(-np.pi, np.pi, 9))
def test_fold_yaw(yaw):
    f = fold_yaw(yaw)
    assert -np.pi / 4 <= f < np.pi / 4 and np.isclose(((yaw - f) / (np.pi / 2)) % 1, 0, atol=1e-9) | \
        np.isclose(((yaw - f) / (np.pi / 2)) % 1, 1, atol=1e-9)


def test_noise_model():
    rng = np.random.default_rng(0)
    rgb, depth = np.full((100, 100, 3), 128, np.uint8), np.ones((100, 100))
    r0, d0 = NoiseModel().apply(rgb, depth, rng)
    assert np.array_equal(r0, rgb) and np.array_equal(d0, depth)
    _, d1 = NoiseModel(depth_a=0.001, depth_b=0.002, depth_dropout=0.1).apply(rgb, depth, rng)
    assert 0.07 < (d1 == 0).mean() < 0.13
    assert 0.002 < d1[d1 > 0].std() < 0.004  # sigma = 0.001 + 0.002 * 1^2


# ---------- M12 vision-driven policy ----------
def test_vision_policy_uses_estimates_not_truth(setup):
    env, _ = setup
    pol = VisionPolicy(ScriptedPickPlace(), noise="realistic", noise_seed=1)
    env.reset(2)
    pol.reset(env)
    assert not np.allclose(pol.view.cube_pose()[:3, 3], env.cube_pose()[:3, 3], atol=1e-9)
    assert np.allclose(pol.view.cube_pose()[:3, 3], pol.cube_est.position)
    assert pol.errors.cube_xy < 0.005


@pytest.mark.parametrize("scene", ["open", "wall"])
def test_vision_driven_pick_place_succeeds(scene):
    from ari.skills.planned_pick_place import PlannedPickPlace

    env = PickPlaceEnv(SceneConfig.with_wall() if scene == "wall" else SceneConfig())
    base = PlannedPickPlace(plan_seed=3) if scene == "wall" else ScriptedPickPlace()
    r = run_episode(env, VisionPolicy(base, noise="realistic", noise_seed=3), 3)
    assert r.success and r.wall_contact_steps == 0, r.failure
