"""M22: one benchmark for every single-object policy in the repository.

Each policy is run on the same 100 seeded episodes under each CONDITION. A condition changes
exactly one thing relative to the scene all learned policies were trained in (the last one
changes two, on purpose). None of the policies was trained or tuned on any of these
conditions, except where noted in docs/results_stage7.md.

All policies act through the same 20 Hz action interface and the same executor
(joint-space interpolation, Stage 5), so differences between rows are differences between
policies.

Seeds: episodes 0-99; sensing noise seed + 100000; cube shift seed + 500000 (as before).
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ari.envs.pick_place import EpisodeSpec, PickPlaceEnv, SceneConfig

TRAIN_CUBE_REGION = SceneConfig().cube_region


@dataclass(frozen=True)
class Condition:
    label: str                              # short name for tables and plots
    what: str                               # one line: what differs from training
    scene: dict = field(default_factory=dict)   # SceneConfig overrides (physics, geometry)
    shift_cube: bool = False                # move the cube 5-10 cm once, 1 s into the episode
    jitter: float = 0.0                     # m: fresh Gaussian error on the sensed cube position, every reading
    offset: float = 0.0                     # m: constant error on the sensed cube position (random direction per episode)
    delay_steps: int = 0                    # actions take effect this many 50 ms steps late
    delay_absolute: bool = False            # experiment: the delayed action keeps the pose target it was computed for
    outside: float = 0.0                    # m: cube starts in a band this wide AROUND the training area


CONDITIONS: dict[str, Condition] = {
    "nominal": Condition("Nominal", "the training conditions"),
    "moved": Condition("Cube moved", "cube moved 5-10 cm, 1 s into the episode", shift_cube=True),
    "jitter": Condition("Sensing jitter", "cube position sensed with 2 mm random error at every step", jitter=0.002),
    "offset": Condition("Sensing offset", "cube position sensed 5 mm off, constant per episode", offset=0.005),
    "latency": Condition("Latency 100 ms", "every action takes effect 100 ms late", delay_steps=2),
    "outside": Condition("Outside area", "cube starts up to 5 cm outside the training area", outside=0.05),
    "heavy": Condition("Heavy cube", "cube mass 0.15 kg instead of 0.1", scene={"cube_mass": 0.15}),
    "small": Condition("3 cm cube", "3 cm cube instead of 4 cm", scene={"cube_half_size": 0.015}),
    "large": Condition("5 cm cube", "5 cm cube instead of 4 cm", scene={"cube_half_size": 0.025}),
    "friction07": Condition("Friction 0.7", "cube friction 0.7 instead of 1.0", scene={"cube_friction": 0.7}),
    "friction05": Condition("Friction 0.5", "cube friction 0.5 instead of 1.0", scene={"cube_friction": 0.5}),
    "stiff": Condition("Stiff gripper", "gripper servo 10x stiffer", scene={"gripper_stiffness_scale": 10.0}),
    "friction05_stiff": Condition("Friction 0.5, stiff", "cube friction 0.5 AND gripper servo 10x stiffer",
                                  scene={"cube_friction": 0.5, "gripper_stiffness_scale": 10.0}),
}

# Severity sweeps: the same perturbation at increasing strength. Cell name "axis@value".
SWEEPS: dict[str, tuple[str, str, list]] = {
    "jitter": ("jitter", "Sensing jitter (mm)", [0.001, 0.002, 0.005, 0.010]),
    "offset": ("offset", "Sensing offset (mm)", [0.002, 0.005, 0.010, 0.020]),
    "latency": ("delay_steps", "Action latency (ms)", [1, 2, 3, 4]),
}


def get_condition(name: str) -> Condition:
    """'jitter' -> the benchmark condition; 'jitter@0.005' -> the same perturbation at another strength."""
    if "@" not in name:
        return CONDITIONS[name]
    axis, value = name.split("@")
    if axis == "latency_abs":                    # experiment (results_stage7.md): same delay, absolute targets
        return replace(CONDITIONS["latency"], delay_steps=int(value), delay_absolute=True,
                       label=f"Latency {50 * int(value)} ms, absolute", what="delayed actions keep their original pose target")
    if axis == "heavy":                          # footnote measurement: another cube mass (kg)
        return replace(CONDITIONS[axis], scene={"cube_mass": float(value)})
    attr = SWEEPS[axis][0]
    return replace(CONDITIONS[axis], **{attr: type(getattr(CONDITIONS[axis], attr))(float(value))})


# Policy specs (see learning.evaluate.make_policy) and how each was obtained.
M = "outputs/models"
POLICIES: dict[str, tuple[str, str]] = {
    "scripted": ("Scripted (open-loop)", "hand-written; reads the cube pose once, then follows a timed trajectory"),
    "reactive": ("Reactive (feedback)", "hand-written; re-reads the scene at every step"),
    f"{M}/bc_reactive_100.pt": ("Clone, 100 demos", "behaviour cloning of the reactive policy"),
    f"{M}/bc_reactive_500.pt": ("Clone, 500 demos", "behaviour cloning of the reactive policy"),
    f"{M}/bc_dagger.pt": ("Clone + DAgger", "500 demos + 3 DAgger rounds; training included moved cubes"),
    f"{M}/rl_residual.pt": ("100-demo clone + RL", "residual PPO on the 100-demo clone; 30% of training episodes had a moved cube"),
}
IMAGE_POLICY = (f"{M}/vision_cpu.pt", "Image policy", "two 64x64 cameras; cloned from the reactive policy, 200 demos")
IMAGE_CONDITIONS = ("nominal", "moved", "outside")   # a short row: each cell renders 2 cameras per step on CPU


class BenchEnv(PickPlaceEnv):
    """PickPlaceEnv plus the perturbations a Condition asks for.

    Sensing errors are applied where every policy reads the cube (`cube_pose`); success is
    always judged on the true simulator state."""

    def __init__(self, cond: Condition):
        super().__init__(SceneConfig(**cond.scene))
        self.cond = cond
        self._noise = np.random.default_rng(0)
        self._bias = np.zeros(2)

    def sample_episode(self, seed: int) -> EpisodeSpec:
        if not self.cond.outside:
            return super().sample_episode(seed)
        rng = np.random.default_rng(seed)
        (cx0, cx1), (cy0, cy1) = TRAIN_CUBE_REGION
        (tx0, tx1), (ty0, ty1) = self.cfg.target_region
        b = self.cond.outside
        while True:
            cube_xy = rng.uniform([cx0 - b, cy0 - b], [cx1 + b, cy1 + b])
            target_xy = rng.uniform([tx0, ty0], [tx1, ty1])
            inside = cx0 <= cube_xy[0] <= cx1 and cy0 <= cube_xy[1] <= cy1
            if not inside and np.linalg.norm(cube_xy - target_xy) >= self.cfg.min_separation:
                return EpisodeSpec(seed, cube_xy, rng.uniform(-np.pi, np.pi), target_xy, 0.0)

    def reset(self, seed: int) -> EpisodeSpec:
        self._noise = np.random.default_rng(seed + 100_000)
        ang = self._noise.uniform(0, 2 * np.pi)
        self._bias = self.cond.offset * np.array([np.cos(ang), np.sin(ang)])
        return super().reset(seed)

    def cube_pose(self) -> np.ndarray:
        T = super().cube_pose()
        if self.cond.jitter or self.cond.offset:
            T[:2, 3] += self._bias + self.cond.jitter * self._noise.standard_normal(2)
        return T


class Delayed:
    """Actions take effect `k` policy steps late. Until the first one arrives the arm holds.

    An action is a step relative to the gripper pose measured when it is APPLIED
    (learning/spaces.py). A late action is therefore added to a pose the policy never saw: the
    commanded target overshoots by however far the arm travelled during the delay.
    env given (absolute=True in the condition): the action is converted to the pose target it
    meant when it was computed, and re-expressed relative to the pose at application time. Same
    delay, no overshoot. Used to separate the two effects (results_stage7.md)."""

    def __init__(self, policy, k: int, env=None):
        self.policy, self.k, self.env = policy, k, env
        self.queue: deque = deque()

    def reset(self):
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.queue = deque(None for _ in range(self.k))

    def __call__(self, obs):
        from ari.learning.spaces import action_from_target, target_from_action

        a = self.policy(obs)
        self.queue.append(target_from_action(self.env.tcp_pose(), a) if self.env is not None else a)
        late = self.queue.popleft()
        if late is None:
            return np.zeros(5, dtype=np.float32)
        return action_from_target(self.env.tcp_pose(), *late) if self.env is not None else late


def _chunk(args):
    spec, cond_name, seeds, max_time = args
    import torch

    from ari.learning import demos
    from ari.learning.evaluate import CubeShift, make_policy

    torch.set_num_threads(1)
    cond = get_condition(cond_name)
    env = BenchEnv(cond)
    if spec == "reactive#size":                 # the reactive policy TOLD the cube size (one parameter)
        from ari.skills.reactive_pick_place import ReactiveExpert

        expert = ReactiveExpert(half=env.cfg.cube_half_size)
        policy = lambda obs: expert(env)        # noqa: E731
    else:
        policy = make_policy(spec, env)
    if cond.delay_steps:
        policy = Delayed(policy, cond.delay_steps, env if cond.delay_absolute else None)
    return [asdict(demos.rollout(env, policy, s, max_time=max_time, joint_interp=True,
                                 disturbance=CubeShift(s) if cond.shift_cube else None)) for s in seeds]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    c = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / (1 + z**2 / n)
    return float(max(0.0, c - h)), float(min(1.0, c + h))


def run_cell(spec: str, cond_name: str, seeds=range(100), workers: int = 2, max_time: float = 20.0) -> dict:
    from ari.learning.evaluate import _pmap

    seeds = list(seeds)
    chunks = [seeds[i::workers * 2] for i in range(workers * 2)]
    rows = sorted((r for c in _pmap(_chunk, [(spec, cond_name, c, max_time) for c in chunks], workers) for r in c),
                  key=lambda r: r["seed"])
    k = sum(r["success"] for r in rows)
    ok = [r for r in rows if r["success"]]
    return {
        "policy": spec, "condition": cond_name, "n": len(rows), "successes": k, "success_rate": k / len(rows),
        "ci95": list(wilson(k, len(rows))),
        "failures": dict(Counter(r["failure"] for r in rows if not r["success"])),
        "median_time_s": float(np.median([r["duration"] for r in ok])) if ok else None,
        "median_placement_mm": float(np.median([r["placement_error"] for r in ok]) * 1000) if ok else None,
        "episodes": [{"seed": r["seed"], "success": bool(r["success"]), "failure": r["failure"],
                      "placement_mm": round(1000 * r["placement_error"], 2), "time_s": round(r["duration"], 2)}
                     for r in rows],
    }
