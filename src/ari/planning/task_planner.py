"""M20: from a multi-step instruction to verified execution.

    text  --split-->  clauses  --parser-->  plan  --validate against the scene-->
          execute each step with a skill  --check the result-->  retry / repair  --> report

Four ideas, each independent of how smart the language component is:

1. A plan is DATA (a list of Step objects), not behaviour. It can be inspected and rejected
   before the robot moves.
2. Validation: never trust the planner. Every step is checked against the actual scene and
   the rest of the plan (is the target free? is the request understood? is it already done?).
   This gate is the same whether the plan comes from a 45-word GRU or a large language model.
3. Post-conditions: "put X on Y" is finished when X is measured to be on Y, not when the
   skill returns. Each step is verified, and at the end ALL steps are verified again, because
   a later action can undo an earlier one.
4. Bounded recovery: a failed check triggers a retry from a known state (the home pose), up
   to a limit; then the executor stops and says what it could not do.

Scene state here comes from the simulator. With Stage 3's perception in its place, the same
checks would run on estimated poses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from ari.envs.multi_object import CUBE_COLORS, MIN_CUBE_SEPARATION, TARGET_COLORS
from ari.learning import demos
from ari.learning.spaces import POLICY_DT
from ari.robot.ik import Q_HOME

COLOR_WORDS = set(CUBE_COLORS) | set(TARGET_COLORS)
_CONNECTIVE = re.compile(r"\s*(?:,?\s*and then\b|,?\s*then\b|,?\s*after that\b,?|,?\s*next\b,?|;|\.)\s*")


@dataclass
class Step:
    cube: str
    target: str
    confidence: float = 1.0
    text: str = ""


def split_clauses(text: str) -> list[str]:
    """Split a multi-step instruction into single 'put X on Y' clauses.

    Splitting on words like "then" is not enough: "grab the red cube then drop it on the blue
    target" is ONE step. A clause is complete once it mentions two colours (a cube and a
    target), so fragments are merged until they do.
    """
    text = re.sub(r"^\s*(first|firstly)\b,?\s*", "", text.lower().strip())
    fragments = [f for f in _CONNECTIVE.split(text) if f and f.strip()]
    clauses, current = [], ""
    for frag in fragments:
        frag = re.sub(r"^(finally|lastly)\b,?\s*", "", frag.strip())
        current = f"{current} then {frag}".strip() if current else frag
        if sum(w in COLOR_WORDS for w in re.findall(r"[a-z]+", current)) >= 2:
            clauses.append(current)
            current = ""
    if current:
        clauses.append(current)      # incomplete: validation will reject it
    return clauses


def make_plan(text: str, parser_model) -> list[Step]:
    from ari.language import parser

    return [Step(*parser.parse(parser_model, clause), text=clause) for clause in split_clauses(text)]


def validate_plan(steps: list[Step], env, min_confidence: float = 0.8) -> list[str]:
    """Returns a list of problems; empty means the plan may be executed.
    Simulates the plan's effect on 'which cube is on which target' to catch conflicts."""
    problems = []
    if not steps:
        return ["no steps were understood"]
    occupied = {t: env.cube_on_target(t) for t in TARGET_COLORS}          # target -> cube index or None
    for k, s in enumerate(steps, 1):
        n_colors = sum(w in COLOR_WORDS for w in re.findall(r"[a-z]+", s.text)) if s.text else 2
        if n_colors < 2:
            problems.append(f"step {k}: '{s.text}' does not name both a cube and a target")
            continue
        if s.confidence < min_confidence:
            problems.append(f"step {k}: unsure what '{s.text}' means (confidence {s.confidence:.2f})")
        ci = CUBE_COLORS.index(s.cube)
        holder = occupied.get(s.target)
        if holder is not None and holder != ci:
            problems.append(f"step {k}: the {s.target} target already holds the {CUBE_COLORS[holder]} cube")
        for t, c in occupied.items():                                      # the cube leaves wherever it was
            if c == ci:
                occupied[t] = None
        occupied[s.target] = ci
    return problems


# ---------------------------------------------------------------- fault injection (for testing recovery)
class Faults:
    """Seeded mishaps: a cube slips out of the gripper while being carried ('drop'), or a cube
    that was already placed gets knocked off its target ('bump')."""

    def __init__(self, seed: int, n_steps: int, p_drop: float = 0.3, p_bump: float = 0.3):
        self.rng = np.random.default_rng(seed + 600_000)
        self.drop_steps = {k for k in range(n_steps) if self.rng.random() < p_drop}
        self.bump_steps = {k for k in range(n_steps - 1) if self.rng.random() < p_bump}
        self.injected = []

    def _free_spot(self, env, moving: int, region, near=None, dist=(0.06, 0.10)):
        for _ in range(200):
            if near is None:
                (x0, x1), (y0, y1) = region
                xy = self.rng.uniform([x0, y0], [x1, y1])
            else:
                ang, d = self.rng.uniform(0, 2 * np.pi), self.rng.uniform(*dist)
                xy = near + d * np.array([np.cos(ang), np.sin(ang)])
            others = [env.cube_position(i)[:2] for i in range(len(CUBE_COLORS)) if i != moving]
            if all(np.linalg.norm(xy - o) >= MIN_CUBE_SEPARATION for o in others) and \
                    all(np.linalg.norm(xy - env.target_position(j)[:2]) > env.cfg.target_radius + 0.03
                        for j in range(len(TARGET_COLORS))) and 0.35 <= xy[0] <= 0.65 and -0.3 <= xy[1] <= 0.3:
                return xy
        return None

    def drop_disturbance(self, step_index: int, env):
        """Returns a per-step callable for run_policy, or None."""
        if step_index not in self.drop_steps:
            return None
        done = [False]

        def disturb(env, t):
            if not done[0] and env.data.xpos[env.cube_body][2] > 0.10:
                xy = self._free_spot(env, env.cube_index, env.cfg.cube_region)
                if xy is not None:
                    env.teleport_cube(env.cube_index, xy)
                    self.injected.append(("drop", step_index))
                done[0] = True

        self.drop_steps.discard(step_index)      # a given step is sabotaged once, not on every retry
        return disturb

    def bump_after(self, step_index: int, env, cube: str) -> None:
        if step_index in self.bump_steps:
            i = CUBE_COLORS.index(cube)
            xy = self._free_spot(env, i, None, near=env.cube_position(i)[:2])
            if xy is not None:
                env.teleport_cube(i, xy)
                self.injected.append(("bump", step_index))
            self.bump_steps.discard(step_index)


# ---------------------------------------------------------------- execution
@dataclass
class TaskResult:
    success: bool                 # every step's post-condition holds at the end (ground truth)
    refused: bool = False         # validation rejected the plan; the robot did not move
    problems: list = field(default_factory=list)
    attempts: int = 0             # skill executions in total
    repairs: int = 0              # steps redone after the final verification found them undone
    faults: list = field(default_factory=list)
    sim_time: float = 0.0
    gave_up: bool = False


def go_home(env, ex: demos.ActionExecutor, duration: float = 1.5, frame_callback=None) -> None:
    """Return to the home posture with the gripper open: every skill then starts from the
    state it was trained and tested in."""
    q0 = env.data.qpos[env.arm_qpos].copy()
    ex.ctrl.set_gripper(False)
    n = int(duration / ex.ctrl.control_dt)
    for i in range(1, n + 1):
        s = i / n
        s = s**3 * (10 - 15 * s + 6 * s * s)                      # minimum-jerk timing (Stage 2)
        ex.ctrl.set_joint_target(q0 + s * (Q_HOME - q0))
        ex.ctrl.step()
        if frame_callback is not None:
            frame_callback(env)
    ex.prev = env.tcp_pose().copy()


class TaskExecutor:
    def __init__(self, env, skill, monitor: bool = True, max_attempts: int = 3, skill_time: float = 15.0,
                 joint_interp: bool = True, fallback=None, frame_callback=None):
        """skill: a rollout policy (obs -> action, optional .reset()) acting on env's focused pair.
        fallback: a second skill used for every attempt after a step's first failure. A learned
        skill only works inside the conditions it was trained on; after a mishap the scene is
        often outside them, and retrying the same skill there does not help (devlog.md)."""
        self.env, self.skill, self.monitor, self.fallback = env, skill, monitor, fallback
        self.max_attempts, self.skill_time, self.joint_interp = max_attempts, skill_time, joint_interp
        self.frame_callback = frame_callback               # called with env during execution (for videos)

    def _attempt(self, step: Step, ex, disturbance, use_fallback: bool = False) -> bool:
        self.env.focus(step.cube, step.target)
        skill = self.fallback if (use_fallback and self.fallback is not None) else self.skill
        if hasattr(skill, "reset"):
            skill.reset()                          # open-loop skills re-read the scene here
        ok, _ = demos.run_policy(self.env, ex, skill, self.skill_time, disturbance, self.frame_callback)
        go_home(self.env, ex, frame_callback=self.frame_callback)
        return ok

    def run(self, steps: list[Step], faults: Faults | None = None, validate: bool = True) -> TaskResult:
        env = self.env
        res = TaskResult(False)
        if validate:
            res.problems = validate_plan(steps, env)
            if res.problems:
                res.refused = True
                return res
        t0 = env.data.time
        ex = demos.ActionExecutor(env, joint_interp=self.joint_interp)
        tries = [0] * len(steps)

        def execute(k):
            dist = faults.drop_disturbance(k, env) if faults is not None else None
            tries[k] += 1
            res.attempts += 1
            self._attempt(steps[k], ex, dist, use_fallback=tries[k] > 1)
            return env.is_placed(steps[k].cube, steps[k].target)      # measured, not assumed

        for k, step in enumerate(steps):
            if self.monitor and env.is_placed(step.cube, step.target):
                continue                                              # already done: nothing to do
            done = execute(k)
            while self.monitor and not done and tries[k] < self.max_attempts:
                done = execute(k)
            if faults is not None:
                faults.bump_after(k, env, step.cube)

        if self.monitor:                                              # final verification of the WHOLE plan
            changed = True
            while changed:
                changed = False
                for k, step in enumerate(steps):
                    if not env.is_placed(step.cube, step.target) and tries[k] < self.max_attempts:
                        res.repairs += 1
                        execute(k)
                        changed = True
        for _ in range(int(0.3 / POLICY_DT)):
            ex.ctrl.step()
        res.success = all(env.is_placed(s.cube, s.target) for s in steps)
        res.gave_up = self.monitor and not res.success
        res.faults = list(faults.injected) if faults is not None else []
        res.sim_time = float(env.data.time - t0)
        return res
