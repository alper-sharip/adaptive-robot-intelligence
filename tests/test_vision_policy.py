"""M15: cameras, the visuomotor network, and the image-policy plumbing (small and CPU-fast)."""

import numpy as np
import pytest
import torch

from ari.envs.pick_place import PickPlaceEnv
from ari.learning import demos, vision
from ari.skills.reactive_pick_place import ReactiveExpert


@pytest.fixture(scope="module")
def env():
    return PickPlaceEnv()


@pytest.fixture(scope="module")
def observer(env):
    return vision.ImageObserver(env)


def red_pixels(img):   # img: (3, H, W) uint8
    r, g, b = img.astype(int)
    return int(((r > 150) & (r - g > 80) & (r - b > 80)).sum())   # lit cube faces are ~(255, 102, 102)


def test_observation_shapes(env, observer):
    env.reset(0)
    im, pr = observer.images(), observer.proprio()
    assert im.shape == (2, 3, vision.IMG, vision.IMG) and im.dtype == np.uint8
    assert pr.shape == (vision.PROPRIO_DIM,) and np.allclose(pr[:3], env.tcp_pose()[:3, 3], atol=1e-6)


def test_front_camera_always_sees_the_cube_at_the_start(env, observer):
    """The wrist camera does NOT always see the cube from the home pose (seed 1: 0 red pixels);
    the front camera must, or the student has no way to know where to go."""
    for seed in range(8):
        env.reset(seed)
        assert red_pixels(observer.images()[0]) >= 4, "front camera lost the cube"


def test_wrist_camera_moves_with_the_hand(env, observer):
    env.reset(0)
    before = observer.images()
    ex = ReactiveExpert()
    demos.rollout(env, lambda o: ex(env), 0, max_time=2.3)   # hand is now just above the cube
    after = observer.images()
    assert red_pixels(after[1]) > 5 * max(1, red_pixels(before[1]))   # cube looms large at the wrist
    assert abs(red_pixels(after[0]) - red_pixels(before[0])) <= 3      # ...and is unchanged for the fixed camera


def test_spatial_softmax_finds_a_bright_spot():
    f = torch.full((1, 1, 11, 11), -20.0)
    f[0, 0, 2, 8] = 20.0                       # row 2 (y), column 8 (x)
    xy = vision.SpatialSoftmax()(f)[0]
    assert torch.allclose(xy, torch.tensor([-1 + 2 * 8 / 10, -1 + 2 * 2 / 10]), atol=1e-4)


def _tiny_dataset(env, observer, n_eps=3):
    ex, I, P, A, E = ReactiveExpert(), [], [], [], []
    for ep in range(n_eps):
        def step(o):
            a = ex(env)
            I.append(observer.images()), P.append(observer.proprio()), A.append(a), E.append(ep)
            return a
        demos.rollout(env, step, 10_000 + ep, max_time=3.0)
    return {"images": np.array(I), "proprio": np.array(P), "action": np.array(A), "episode_index": np.array(E)}


def test_fit_reduces_loss_and_policy_roundtrips(env, observer, tmp_path):
    data = _tiny_dataset(env, observer)
    net, stats, hist = vision.fit(data, epochs=12, batch_size=16, val_fraction=0.34,
                                  device=torch.device("cpu"), verbose=False)
    assert hist[-1]["train_motion"] < 0.8 * hist[0]["train_motion"]
    vision.save(tmp_path / "v.pt", net, stats)
    env.reset(1)
    a1 = vision.ImagePolicy(env, net, stats)()
    a2 = vision.ImagePolicy(env, *vision.load(tmp_path / "v.pt"))()
    assert a1.shape == (5,) and np.allclose(a1, a2, atol=1e-6) and a1[4] in (0.0, 1.0)


def test_make_policy_recognises_a_vision_checkpoint(env, tmp_path):
    from ari.learning.evaluate import make_policy

    vision.save(tmp_path / "v.pt", vision.VisuomotorNet(), {
        "prop_mean": torch.zeros(6), "prop_std": torch.ones(6), "act_mean": torch.zeros(4), "act_std": torch.ones(4)})
    assert isinstance(make_policy(str(tmp_path / "v.pt"), env), vision.ImagePolicy)


def test_student_never_receives_object_positions(env):
    """The image policy's inputs are pixels and the gripper's own pose: nothing else."""
    pol = vision.ImagePolicy(env, vision.VisuomotorNet(), {
        "prop_mean": torch.zeros(6), "prop_std": torch.ones(6), "act_mean": torch.zeros(4), "act_std": torch.ones(4)})
    env.reset(2)
    a = pol(np.full(20, np.nan, np.float32))    # poison the state observation: must be ignored
    assert np.isfinite(a).all()


def test_shipped_image_policy_grasps_from_pixels(env):
    """The CPU-scale image student (43% full-task success) reliably lifts the cube; check one
    episode it is known to complete."""
    from pathlib import Path

    from ari.learning.evaluate import make_policy

    path = Path(__file__).resolve().parents[1] / "outputs" / "models" / "vision_cpu.pt"
    if not path.exists():
        pytest.skip("trained model not present")
    r = demos.rollout(env, make_policy(str(path), env), 36)
    assert r.max_cube_height > 0.10 and r.success
