"""Stage 6 evaluation: seeded instruction-following episodes, run in parallel workers.

M19 (one instruction per episode), systems:
    language_policy   the end-to-end network (text + scene -> actions)
    parser+skill      GRU parser picks the cube and target; a Stage 4 skill does the motion
    oracle+expert     told the right cube and target; reactive expert (upper bound)

M20 (two-step instructions, optional injected faults): parser + validation + skill + monitoring.

An episode counts as a success only if the INSTRUCTED cube ends on the INSTRUCTED target.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from ari.envs.multi_object import CUBE_COLORS, TARGET_COLORS
from ari.language import grammar

CONNECTIVES = [", then ", " and then ", "; next ", ". after that ", " then "]
ALL_TEMPLATES = grammar.TRAIN_TEMPLATES + grammar.HELD_OUT_TEMPLATES


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return c - h, c + h


# ---------------------------------------------------------------- M19
def m19_instruction(seed: int):
    """Episode seed -> (split, text, cube, target). A third of the episodes per split."""
    split = ("seen", "new phrasing", "new pairing")[seed % 3]
    text, cube, target = grammar.sample(split, np.random.default_rng(seed + 300_000))
    return split, text, cube, target


def _m19_chunk(args):
    system, seeds, models = args
    import torch

    from ari.envs.multi_object import MultiObjectEnv
    from ari.language import parser, policy
    from ari.learning import demos
    from ari.learning.evaluate import make_policy

    torch.set_num_threads(1)
    env = MultiObjectEnv()
    net = policy.load(models["policy"]) if system == "language_policy" else None
    pm = parser.load(models["parser"]) if system == "parser+skill" else None
    skill = make_policy(models["skill"], env) if system == "parser+skill" else None
    expert = make_policy("reactive", env) if system == "oracle+expert" else None
    out = []
    for seed in seeds:
        split, text, cube, target = m19_instruction(seed)
        env.reset(seed)
        ex = demos.ActionExecutor(env, joint_interp=True)
        if system == "language_policy":
            pol = policy.InstructedPolicy(env, net, text)
            understood = (CUBE_COLORS[pol.attended[0]], TARGET_COLORS[pol.attended[1]])
            env.focus(cube, target)                       # only so run_policy can detect success
        elif system == "parser+skill":
            c, t, _ = parser.parse(pm, text)
            understood, pol = (c, t), skill
            env.focus(c, t)
        else:
            understood, pol = (cube, target), expert
            env.focus(cube, target)
        demos.run_policy(env, ex, pol, max_time=15.0)
        ok = env.is_placed(cube, target) and env.finger_width() > 0.07
        out.append({"seed": seed, "split": split, "text": text, "success": bool(ok),
                    "understood": understood == (cube, target)})
    return out


def run_m19(system: str, models: dict, n: int = 150, workers: int = 2) -> dict:
    from ari.learning.evaluate import _pmap

    seeds = list(range(n))
    rows = [r for c in _pmap(_m19_chunk, [(system, seeds[i::workers * 2], models) for i in range(workers * 2)], workers)
            for r in c]
    rows.sort(key=lambda r: r["seed"])
    summary = {"system": system, "episodes": rows, "by_split": {}}
    for split in ("seen", "new phrasing", "new pairing"):
        sub = [r for r in rows if r["split"] == split]
        k = sum(r["success"] for r in sub)
        summary["by_split"][split] = {"n": len(sub), "success": k / len(sub), "ci95": wilson(k, len(sub)),
                                      "understood": float(np.mean([r["understood"] for r in sub]))}
    return summary


# ---------------------------------------------------------------- M20
def m20_task(seed: int):
    """Episode seed -> (instruction text, intended [(cube, target), (cube, target)])."""
    rng = np.random.default_rng(seed + 400_000)
    cubes = rng.permutation(len(CUBE_COLORS))[:2]
    targets = rng.permutation(len(TARGET_COLORS))[:2]
    intended = [(CUBE_COLORS[c], TARGET_COLORS[t]) for c, t in zip(cubes, targets)]
    clauses = [grammar.render(ALL_TEMPLATES[rng.integers(len(ALL_TEMPLATES))], c, t, rng) for c, t in intended]
    text = ("first " if rng.random() < 0.3 else "") + clauses[0] + CONNECTIVES[rng.integers(len(CONNECTIVES))] + clauses[1]
    return text, intended


def _m20_chunk(args):
    skill_spec, monitor, with_faults, seeds, parser_path, fallback_spec = args
    import torch

    from ari.envs.multi_object import MultiObjectEnv
    from ari.language import parser
    from ari.learning.evaluate import make_policy
    from ari.planning.task_planner import Faults, TaskExecutor, make_plan

    torch.set_num_threads(1)
    env = MultiObjectEnv()
    pm = parser.load(parser_path)
    skill = make_policy(skill_spec, env)
    fallback = make_policy(fallback_spec, env) if fallback_spec else None
    out = []
    for seed in seeds:
        text, intended = m20_task(seed)
        env.reset(seed)
        steps = make_plan(text, pm)
        plan_ok = [(s.cube, s.target) for s in steps] == intended
        res = TaskExecutor(env, skill, monitor=monitor, fallback=fallback).run(
            steps, Faults(seed, len(steps)) if with_faults else None)
        # Judge against what was ASKED, not against the plan the system made for itself.
        done = all(env.is_placed(c, t) for c, t in intended)
        out.append({"seed": seed, "text": text, "plan_ok": plan_ok, "success": bool(done), "refused": res.refused,
                    "attempts": res.attempts, "repairs": res.repairs, "faults": len(res.faults),
                    "sim_time": res.sim_time, "gave_up": res.gave_up})
    return out


def run_m20(skill_spec: str, monitor: bool, with_faults: bool, parser_path: str, n: int = 100, workers: int = 2,
            fallback_spec: str | None = None) -> dict:
    from ari.learning.evaluate import _pmap

    seeds = list(range(n))
    jobs = [(skill_spec, monitor, with_faults, seeds[i::workers * 2], parser_path, fallback_spec)
            for i in range(workers * 2)]
    rows = sorted((r for c in _pmap(_m20_chunk, jobs, workers) for r in c), key=lambda r: r["seed"])
    k = sum(r["success"] for r in rows)
    faulted = [r for r in rows if r["faults"] > 0]
    return {"skill": skill_spec, "fallback": fallback_spec, "monitor": monitor, "faults": with_faults, "n": n,
            "success": k / n,
            "ci95": wilson(k, n), "plan_ok": float(np.mean([r["plan_ok"] for r in rows])),
            "refused": int(sum(r["refused"] for r in rows)),
            "episodes_with_faults": len(faulted),
            "success_when_faulted": float(np.mean([r["success"] for r in faulted])) if faulted else None,
            "mean_attempts": float(np.mean([r["attempts"] for r in rows])),
            "mean_sim_time": float(np.mean([r["sim_time"] for r in rows])),
            "fault_counts": dict(Counter(r["faults"] for r in rows)), "episodes": rows}
