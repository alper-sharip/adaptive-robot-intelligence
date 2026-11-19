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

from collections import Counter
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
    interface: str = "observed"             # what an action's step is relative to (learning/demos.py ActionExecutor)
    camera: bool = False                    # cube and target from cameras (perception/tracking.py), not the simulator
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
    "latency": ("delay_steps", "Action latency (ms)", [1, 2, 3, 4, 6, 10]),
}


def get_condition(name: str) -> Condition:
    """'jitter' -> the benchmark condition; 'jitter@0.005' -> the same perturbation at another strength."""
    if name.startswith("cam:"):                 # the same condition, with the camera loop closed
        base = get_condition(name[4:])
        return replace(base, camera=True, label=f"{base.label}, camera", what=f"{base.what}; cube and target from cameras")
    if "@" not in name:
        return CONDITIONS[name]
    axis, value = name.split("@")
    if axis == "latency_legacy":                 # the action interface before the latency fix (Stages 4-7)
        return replace(CONDITIONS["latency"], delay_steps=int(value), interface="applied",
                       label=f"Latency {50 * int(value)} ms, old interface",
                       what="steps relative to the pose when the action is applied (before the fix)")
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
# The camera loop (perception/tracking.py): conditions where perception matters. Sensing jitter
# and offset are left out: they model sensing errors, and the camera is now the sensor.
CAMERA_CONDITIONS = ("nominal", "moved", "outside", "small", "large", "latency")
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
        self.tracker = None
        if cond.camera:
            from ari.perception.tracking import CubeTracker

            self.tracker = CubeTracker(self)

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
        ep = super().reset(seed)
        if self.tracker is not None:
            self.tracker.reset()
        return ep

    # Success is always judged on the truth, whatever the policies are told.
    def placement_error(self) -> float:
        return float(np.linalg.norm(self.data.xpos[self.cube_body][:2] - self.data.site_xpos[self.target_site][:2]))

    def cube_yaw(self) -> float:
        if self.tracker is not None:
            return self.tracker.cube_yaw()
        return super().cube_yaw()

    def target_pos(self) -> np.ndarray:
        if self.tracker is not None and self.tracker.target is not None:
            return self.tracker.target_pos()
        return super().target_pos()

    def cube_pose(self) -> np.ndarray:
        if self.tracker is not None:
            return self.tracker.cube_pose()
        T = super().cube_pose()
        if self.cond.jitter or self.cond.offset:
            T[:2, 3] += self._bias + self.cond.jitter * self._noise.standard_normal(2)
        return T


def _chunk(args):
    spec, cond_name, seeds, max_time = args
    import torch

    from ari.learning import demos
    from ari.learning.evaluate import CubeShift, make_policy

    torch.set_num_threads(1)
    cond = get_condition(cond_name)
    env = BenchEnv(cond)
    if spec == "reactive#4cm":                  # the reactive policy before the fix: cube size fixed at 4 cm
        from ari.skills.reactive_pick_place import ReactiveExpert

        expert = ReactiveExpert(half=0.02)
        policy = lambda obs: expert(env)        # noqa: E731
    else:
        policy = make_policy(spec, env)
    # Latency sits between the policy and the arm, so it is part of the executor.
    rows = []
    for s in seeds:
        r = asdict(demos.rollout(env, policy, s, max_time=max_time, joint_interp=True,
                                 latency_steps=cond.delay_steps, interface=cond.interface,
                                 disturbance=CubeShift(s) if cond.shift_cube else None))
        if env.tracker is not None:
            r["tracking"] = env.tracker.stats.summary()
        rows.append(r)
    return rows


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
                      "placement_mm": round(1000 * r["placement_error"], 2), "time_s": round(r["duration"], 2),
                      **({"tracking": r["tracking"]} if "tracking" in r else {})}
                     for r in rows],
    }
