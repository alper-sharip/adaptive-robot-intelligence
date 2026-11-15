"""M22: the benchmark conditions do what their descriptions say, and nothing else."""

import numpy as np
import pytest

from ari import benchmark as B
from ari.envs.pick_place import PickPlaceEnv, SceneConfig


def test_every_condition_differs_from_training_in_the_stated_way():
    base = SceneConfig()
    changed = {name: (c.scene, c.shift_cube, c.jitter, c.offset, c.delay_steps, c.outside) for name, c in B.CONDITIONS.items()}
    assert changed["nominal"] == ({}, False, 0.0, 0.0, 0, 0.0)
    for name, c in B.CONDITIONS.items():
        knobs = [bool(c.scene), c.shift_cube, c.jitter > 0, c.offset > 0, c.delay_steps > 0, c.outside > 0]
        assert sum(knobs) == (0 if name == "nominal" else 1), name       # one kind of change per condition
        for key, value in c.scene.items():
            assert getattr(base, key) != value, (name, key)              # and it really is a change
    assert len(B.CONDITIONS["friction05_stiff"].scene) == 2               # the one deliberate double change


def test_sweep_names_resolve_to_the_same_perturbation_at_another_strength():
    assert B.get_condition("jitter") is B.CONDITIONS["jitter"]
    assert B.get_condition("jitter@0.005").jitter == 0.005
    c = B.get_condition("latency@3")
    assert c.delay_steps == 3 and isinstance(c.delay_steps, int) and c.jitter == 0.0
    assert B.get_condition("heavy@0.2").scene == {"cube_mass": 0.2}


def test_outside_condition_starts_the_cube_outside_the_training_area_but_near_it():
    env = B.BenchEnv(B.CONDITIONS["outside"])
    (x0, x1), (y0, y1) = B.TRAIN_CUBE_REGION
    b = B.CONDITIONS["outside"].outside
    for seed in range(50):
        ep = env.sample_episode(seed)
        x, y = ep.cube_xy
        assert not (x0 <= x <= x1 and y0 <= y <= y1)
        assert x0 - b <= x <= x1 + b and y0 - b <= y <= y1 + b
        assert np.linalg.norm(ep.cube_xy - ep.target_xy) >= env.cfg.min_separation
    nominal = B.BenchEnv(B.CONDITIONS["nominal"])
    assert np.allclose(nominal.sample_episode(3).cube_xy, PickPlaceEnv().sample_episode(3).cube_xy)   # unchanged elsewhere


@pytest.mark.parametrize("name,expected", [("offset", 0.005), ("jitter", None)])
def test_sensing_errors_change_what_policies_read_not_what_success_is_judged_on(name, expected):
    env = B.BenchEnv(B.CONDITIONS[name])
    env.reset(4)
    truth = env.data.xpos[env.cube_body][:2].copy()
    reads = np.array([env.cube_pose()[:2, 3] - truth for _ in range(400)])
    err = np.linalg.norm(reads, axis=1)
    if expected is not None:                      # constant offset: same error every time, of the stated size
        assert np.allclose(err, expected, atol=1e-9) and np.allclose(reads, reads[0])
    else:                                         # jitter: zero mean, the stated spread per axis
        assert abs(reads.mean()) < 0.0003 and reads.std() == pytest.approx(0.002, rel=0.15)
    assert np.allclose(env.data.xpos[env.cube_body][:2], truth)          # the cube itself did not move
    assert env.placement_error() == pytest.approx(np.linalg.norm(truth - env.target_pos()[:2]))
    env.reset(4)
    again = env.cube_pose()[:2, 3] - truth
    assert np.allclose(again, reads[0])           # same seed, same sensing error: episodes are reproducible


def test_delayed_policy_shifts_actions_by_k_steps_and_holds_still_first():
    actions = [np.full(5, float(i), dtype=np.float32) for i in range(1, 6)]
    it = iter(actions)
    d = B.Delayed(lambda obs: next(it), 2)
    d.reset()
    out = [d(None) for _ in range(5)]
    assert np.all(out[0] == 0) and np.all(out[1] == 0)
    assert all(np.array_equal(out[i + 2], actions[i]) for i in range(3))


def test_run_cell_reports_counts_and_intervals():
    r = B.run_cell("reactive", "nominal", seeds=range(3), workers=1)
    assert r["n"] == 3 and r["successes"] == 3 and r["success_rate"] == 1.0 and r["failures"] == {}
    assert r["ci95"][0] < 1.0 and r["ci95"][1] == pytest.approx(1.0)
    assert [e["seed"] for e in r["episodes"]] == [0, 1, 2] and all(e["placement_mm"] < 30 for e in r["episodes"])


def test_a_known_fragility_is_reproduced_by_the_harness():
    """The open-loop scripted policy reads the cube once: moving the cube must defeat it."""
    r = B.run_cell("scripted", "moved", seeds=range(3), workers=1)
    assert r["successes"] == 0 and set(r["failures"]) == {"never_lifted"}
