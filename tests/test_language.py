"""Stage 6: multi-object scene, instruction grammar, parsers, language policy, task planner."""

from pathlib import Path

import numpy as np
import pytest
import torch

from ari.envs.multi_object import CUBE_COLORS, MIN_CUBE_SEPARATION, TARGET_COLORS, MultiObjectEnv
from ari.language import grammar, parser, policy
from ari.learning.evaluate import make_policy
from ari.planning.task_planner import Faults, Step, TaskExecutor, make_plan, split_clauses, validate_plan

MODELS = Path(__file__).resolve().parents[1] / "outputs" / "models"


@pytest.fixture(scope="module")
def env():
    return MultiObjectEnv()


@pytest.fixture(scope="module")
def gru():
    path = MODELS / "language_parser.pt"
    return parser.load(path) if path.exists() else parser.train(parser.GRUParser(), steps=1200)


# ---------- scene ----------
def test_scene_sampling_rules(env):
    for seed in range(60):
        ep = env.sample_episode(seed)
        d = np.linalg.norm(ep.cube_xy[:, None] - ep.cube_xy[None], axis=-1) + np.eye(3)
        assert d.min() >= MIN_CUBE_SEPARATION
        # regression: the learned skill failed when a cube was closer to its target than in training
        assert np.linalg.norm(ep.cube_xy[:, None] - ep.target_xy[None], axis=-1).min() >= env.cfg.min_separation


def test_focus_redirects_the_single_object_interface(env):
    ep = env.reset(4)
    for i, c in enumerate(CUBE_COLORS):
        for j, t in enumerate(TARGET_COLORS):
            env.focus(c, t)
            assert np.allclose(env.cube_pose()[:2, 3], ep.cube_xy[i]) and np.allclose(env.target_pos()[:2], ep.target_xy[j])


def test_is_placed_and_occupancy(env):
    env.reset(4)
    assert not env.is_placed("blue", "green") and env.cube_on_target("green") is None
    env.teleport_cube(CUBE_COLORS.index("blue"), env.target_position(TARGET_COLORS.index("green"))[:2])
    assert env.is_placed("blue", "green") and not env.is_placed("red", "green")
    assert env.cube_on_target("green") == CUBE_COLORS.index("blue")


def test_stage4_skill_runs_unchanged_in_the_new_scene(env):
    from ari.learning import demos

    env.reset(2)
    env.focus("yellow", "red")
    ok, _ = demos.run_policy(env, demos.ActionExecutor(env, joint_interp=True), make_policy("reactive", env), 15.0)
    assert ok and env.is_placed("yellow", "red")
    assert not env.is_placed("red", "red") and not env.is_placed("blue", "red")


# ---------- grammar ----------
def test_splits_are_disjoint_and_keep_both_word_orders_in_training():
    assert not set(grammar.TRAIN_TEMPLATES) & set(grammar.HELD_OUT_TEMPLATES)
    assert not set(grammar.TRAIN_PAIRS) & set(grammar.HELD_OUT_PAIRS)
    # regression: holding one of these out lets a model memorize instead of reading word order
    assert ("red", "blue") in grammar.TRAIN_PAIRS and ("blue", "red") in grammar.TRAIN_PAIRS


def test_tokenizer():
    ids = grammar.encode("Relocate the BLUE cube, to the red target!")
    assert ids[0] == grammar.UNK and ids.shape == (grammar.MAX_LEN,) and ids[-1] == grammar.PAD
    assert ids[2] == grammar.VOCAB["blue"]


# ---------- parsers ----------
def test_gru_reads_word_order_and_bag_of_words_cannot(gru):
    assert parser.accuracy(gru, "order only", n=500) > 0.97
    assert parser.accuracy(gru, "new phrasing", n=500) > 0.9
    bow = parser.train(parser.BagOfWords(), steps=600)
    assert parser.accuracy(bow, "order only", n=500) < 0.7
    assert parser.parse(gru, "put the blue cube on the red target")[:2] == ("blue", "red")
    assert parser.parse(gru, "put the red cube on the blue target")[:2] == ("red", "blue")


# ---------- language policy ----------
def test_attention_starts_near_uniform():
    """Regression: a saturated start left the attention confidently and permanently wrong."""
    rng = np.random.default_rng(0)
    tok = torch.as_tensor(np.stack([grammar.encode(grammar.sample("seen", rng)[0]) for _ in range(200)]))
    ct, tt = torch.as_tensor(policy.CUBE_TAGS).repeat(200, 1), torch.as_tensor(policy.TARGET_TAGS).repeat(200, 1)
    for seed in range(3):
        torch.manual_seed(seed)
        ac, at = policy.LanguagePolicy().attention(tok, ct, tt)
        assert torch.cat([ac, at]).max(1).values.mean() < 0.5


def test_policy_does_not_depend_on_slot_order():
    """Shuffling the objects (with their colour tags) must not change the action."""
    torch.manual_seed(0)
    net = policy.LanguagePolicy().eval()
    tok = torch.as_tensor(grammar.encode("put the blue cube on the red target")).unsqueeze(0)
    prop, cubes, targets = torch.randn(1, 6), torch.randn(1, 3, 5), torch.randn(1, 3, 2)
    ct, tt = torch.as_tensor(policy.CUBE_TAGS).unsqueeze(0), torch.as_tensor(policy.TARGET_TAGS).unsqueeze(0)
    a, *_ = net(tok, prop, cubes, targets, ct, tt)
    perm = torch.tensor([2, 0, 1])
    b, *_ = net(tok, prop, cubes[:, perm], targets[:, perm], ct[:, perm], tt[:, perm])
    assert torch.allclose(a, b, atol=1e-5)


def test_shipped_language_policy_grounds_instructions():
    path = MODELS / "language_policy.pt"
    if not path.exists():
        pytest.skip("trained model not present")
    net = policy.load(path)
    env = MultiObjectEnv()
    for cube, target in (("red", "blue"), ("blue", "red"), ("yellow", "red")):
        pol = policy.InstructedPolicy(env, net, f"put the {cube} cube on the {target} target")
        assert (CUBE_COLORS[pol.attended[0]], TARGET_COLORS[pol.attended[1]]) == (cube, target)


# ---------- task planner ----------
@pytest.mark.parametrize("text,n", [
    ("put the red cube on the green target, then the yellow block on the blue zone", 2),
    ("grab the red cube then drop it on the blue target", 1),              # 'then' inside ONE step
    ("first grab the blue box then drop it on the red spot; next move yellow cube to the green area", 2),
    ("onto the blue circle goes the red brick and then the blue cube belongs on the red target. "
     "finally relocate the yellow cube to green target", 3),
])
def test_split_clauses(text, n):
    assert len(split_clauses(text)) == n


def test_plan_matches_instruction(gru):
    steps = make_plan("first grab the blue box then drop it on the red spot; next move yellow cube to the green area", gru)
    assert [(s.cube, s.target) for s in steps] == [("blue", "red"), ("yellow", "green")]


def test_validation_rejects_bad_plans(env, gru):
    env.reset(0)
    assert validate_plan(make_plan("put the red cube on the green target then the yellow cube on the blue zone", gru), env) == []
    two_on_one = validate_plan(make_plan("put the red cube on the green target then the blue cube on the green target", gru), env)
    assert len(two_on_one) == 1 and "already holds" in two_on_one[0]
    assert validate_plan(make_plan("put the red cube somewhere nice", gru), env)
    assert validate_plan([], env)
    assert validate_plan([Step("red", "green", confidence=0.4, text="red thing green thing")], env)
    # moving a cube away first frees its target for another cube
    env.teleport_cube(0, env.target_position(2)[:2])                      # red cube now on the green target
    assert validate_plan([Step("blue", "green", text="blue cube green target")], env)
    assert validate_plan([Step("red", "red", text="red cube red target"),
                          Step("blue", "green", text="blue cube green target")], env) == []


def test_refused_plan_does_not_move_the_robot(env):
    env.reset(0)
    q0 = env.data.qpos.copy()
    res = TaskExecutor(env, make_policy("reactive", env)).run([Step("red", "green", text="red cube"), ])
    assert res.refused and not res.success and np.array_equal(env.data.qpos, q0)


def test_monitoring_repairs_injected_faults(env):
    steps = [Step("red", "green", text="red cube green target"), Step("yellow", "blue", text="yellow cube blue target")]
    skill = make_policy("reactive", env)
    env.reset(1)
    blind = TaskExecutor(env, skill, monitor=False).run(steps, Faults(1, 2, p_drop=0.0, p_bump=1.0))
    assert blind.faults == [("bump", 0)] and not blind.success            # nobody noticed the knocked-off cube
    env.reset(1)
    watched = TaskExecutor(env, skill, monitor=True).run(steps, Faults(1, 2, p_drop=0.0, p_bump=1.0))
    assert watched.success and watched.repairs == 1


def test_open_loop_skill_recovers_through_retries(env):
    steps = [Step("blue", "red", text="blue cube red target")]
    env.reset(3)
    res = TaskExecutor(env, make_policy("scripted", env), monitor=True).run(steps, Faults(3, 1, p_drop=1.0, p_bump=0.0))
    assert res.faults == [("drop", 0)] and res.success and res.attempts == 2


def test_fallback_skill_takes_over_after_a_failed_attempt(env):
    """A skill that never moves fails its first attempt; the retry goes to the fallback."""
    steps = [Step("red", "blue", text="red cube blue target")]

    def idle(_obs):
        return np.zeros(5, dtype=np.float32)

    env.reset(2)
    alone = TaskExecutor(env, idle, monitor=True, max_attempts=2, skill_time=3.0).run(steps)
    assert not alone.success and alone.attempts == 2 and alone.gave_up
    env.reset(2)
    backed = TaskExecutor(env, idle, monitor=True, skill_time=8.0, fallback=make_policy("reactive", env)).run(steps)
    assert backed.success and backed.attempts == 2
