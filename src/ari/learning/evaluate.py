"""Evaluate any policy on seeded episodes (optionally in parallel, optionally with a disturbance),
and DAgger data collection.

Policies are named by a spec string so worker processes can build their own copy:
    "scripted"        the clock-driven Stage 2 expert, run through the 20 Hz action interface
    "reactive"        the state-feedback expert
    "path/to/x.pt"    a trained BCPolicy
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from multiprocessing import get_context

import numpy as np

from ari.envs.pick_place import PickPlaceEnv
from ari.learning import demos
from ari.learning.spaces import action_from_target
from ari.skills.pick_place import ScriptedPickPlace
from ari.skills.reactive_pick_place import ReactiveExpert


class ScriptedAsPolicy:
    """Adapter: the clock-driven expert as an obs -> action callable (it ignores obs)."""

    def __init__(self, env):
        self.env, self.expert = env, ScriptedPickPlace()

    def reset(self):
        self.expert.reset(self.env)   # reads the cube pose ONCE, here

    def __call__(self, obs):
        a = None
        for _ in range(5):
            a = self.expert.act(self.env, 0.01)
            if a is None:
                return np.zeros(5, dtype=np.float32)
        return action_from_target(self.env.tcp_pose(), a[0], a[1])


def make_policy(spec: str, env):
    if spec == "scripted":
        return ScriptedAsPolicy(env)
    if spec == "reactive":
        ex = ReactiveExpert()
        return lambda obs: ex(env)
    from ari.learning.bc import BCPolicy

    return BCPolicy.load(spec)


class CubeShift:
    """Disturbance: once, at time `at` (the gripper is then on its way to the cube), move the
    cube 5-10 cm in a random direction (kept inside the cube sampling region)."""

    def __init__(self, seed: int, at: float = 1.0):
        self.rng, self.at, self.done = np.random.default_rng(seed + 500_000), at, False
        self.applied = False

    def __call__(self, env, t):
        if self.done or t < self.at:
            return
        self.done = True
        if env.data.xpos[env.cube_body][2] > env.cfg.cube_half_size + 0.005:
            return  # already lifted: leave it
        (x0, x1), (y0, y1) = env.cfg.cube_region
        xy = env.data.xpos[env.cube_body][:2].copy()
        for _ in range(100):
            ang, dist = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(0.05, 0.10)
            new = xy + dist * np.array([np.cos(ang), np.sin(ang)])
            if x0 <= new[0] <= x1 and y0 <= new[1] <= y1 and \
                    np.linalg.norm(new - env.target_pos()[:2]) > env.cfg.min_separation:
                demos.move_cube(env, new)
                self.applied = True
                return


def _eval_chunk(args):
    spec, seeds, disturb, max_time = args
    import torch

    torch.set_num_threads(1)
    env = PickPlaceEnv()
    policy = make_policy(spec, env)
    return [demos.rollout(env, policy, s, max_time=max_time,
                          disturbance=CubeShift(s) if disturb else None) for s in seeds]


def drop_repeats(obs: np.ndarray, act: np.ndarray, tol: float = 1e-3):
    """Keep a sample only if the state moved by more than `tol` (1 mm / 0.001 in any entry)
    since the last kept sample.

    When the learner gets stuck, the rest of the rollout is ONE state recorded hundreds of
    times. Left in, those copies swamp the dataset: starting DAgger from a weak policy (71%)
    made it worse every iteration, down to 25% (devlog.md).
    """
    keep, last = [], None
    for i, o in enumerate(obs):
        if last is None or np.abs(o - last).max() > tol:
            keep.append(i)
            last = o
    return obs[keep], act[keep]


def _dagger_chunk(args):
    spec, seeds, max_time, dedupe = args
    import torch

    torch.set_num_threads(1)
    env = PickPlaceEnv()
    policy, expert = make_policy(spec, env), ReactiveExpert()
    O, A, ok = [], [], 0
    for s in seeds:
        obs, act = [], []

        def learner_with_expert_labels(o):
            obs.append(o)
            act.append(expert(env))     # what the expert WOULD do here...
            return policy(o)            # ...but the learner's action is what gets executed

        ok += demos.rollout(env, learner_with_expert_labels, s, max_time=max_time).success
        o, a = np.array(obs), np.array(act)
        if dedupe:
            o, a = drop_repeats(o, a)
        O.append(o), A.append(a)
    return O, A, ok


def _pmap(fn, jobs, workers):
    if workers <= 1:
        return [fn(j) for j in jobs]
    with get_context("spawn").Pool(workers) as pool:
        return pool.map(fn, jobs)


def evaluate(spec: str, seeds=range(100), disturb: bool = False, workers: int = 2, max_time: float = 25.0):
    seeds = list(seeds)
    chunks = [seeds[i::workers * 2] for i in range(workers * 2)]
    results = [r for c in _pmap(_eval_chunk, [(spec, c, disturb, max_time) for c in chunks], workers) for r in c]
    results.sort(key=lambda r: r.seed)
    n, k = len(results), sum(r.success for r in results)
    p, z = k / n, 1.96
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    ok = [r for r in results if r.success]
    return {
        "policy": spec, "disturbed": disturb, "n": n, "success_rate": p, "ci95": [c - h, c + h],
        "failures": dict(Counter(r.failure for r in results if not r.success)),
        "median_time_s": float(np.median([r.duration for r in ok])) if ok else None,
        "median_placement_mm": float(np.median([r.placement_error for r in ok]) * 1000) if ok else None,
        "episodes": [asdict(r) for r in results],
    }


def dagger_collect(spec: str, seeds, workers: int = 2, max_time: float = 15.0, dedupe: bool = True):
    """Roll out the LEARNER on `seeds`; label every visited state with the expert's action.
    Returns (obs, act, episode_lengths, learner_success_count)."""
    seeds = list(seeds)
    chunks = [seeds[i::workers * 2] for i in range(workers * 2)]
    out = _pmap(_dagger_chunk, [(spec, c, max_time, dedupe) for c in chunks], workers)
    O = [o for chunk in out for o in chunk[0]]
    A = [a for chunk in out for a in chunk[1]]
    return O, A, sum(chunk[2] for chunk in out)


def summary_line(r: dict) -> str:
    return (f"{r['policy']:<38s} {'disturbed' if r['disturbed'] else 'nominal':9s} "
            f"{100 * r['success_rate']:5.1f}%  (95% CI {100 * r['ci95'][0]:.1f}-{100 * r['ci95'][1]:.1f})  "
            f"time {r['median_time_s'] or float('nan'):.1f}s  place {r['median_placement_mm'] or float('nan'):.1f}mm  "
            f"failures {r['failures'] or 'none'}")
