# Stage 3 results: vision-driven pick-and-place

The robot takes one RGB-D snapshot at the start of each episode (arm at home), estimates the cube
pose and the target position, and runs the unchanged Stage 2 / M8 policies on those estimates.
Same 100 seeded episodes as before. Perception errors are measured against simulator truth for
every episode (the policy never sees the truth).

Reproduce: `python scripts/evaluate_vision.py --scene <open|wall> --noise <none|realistic|poor>`
(and `--perception gt` for the ground-truth reference).

## Task success

| Scene / policy | Ground truth | Vision, no noise | Vision, realistic noise | Vision, poor noise |
|---|---|---|---|---|
| Open table / scripted | 100% (96.3-100) | 100% (96.3-100) | 100% (96.3-100) | 100% (96.3-100) |
| Wall / planned (clean success) | 96% (90.2-98.4) | 96% (90.2-98.4) | 98% (93.0-99.4) | 98% (93.0-99.4) |
| Median placement error, open | 1.9 mm | 2.3 mm | 2.8 mm | 3.0 mm |

All 600 vision episodes detected both the cube and the target. Every wall-scene failure is a
planner refusal (`no_grasp_plan` / `no_place_plan`), never a collision, and never a perception failure.

## Perception accuracy (open scene, 100 episodes; median / max)

| Noise | Cube xy | Cube height bias | Cube yaw | Target xy |
|---|---|---|---|---|
| none | 0.29 / 0.71 mm | −0.41 mm | 0.00 / 1.30° | 1.10 / 2.09 mm |
| realistic | 0.58 / 1.79 mm | −0.45 mm | 0.66 / 3.70° | 1.04 / 1.54 mm |
| poor | 0.57 / 2.56 mm | +0.25 mm | 0.92 / 8.73° | 1.89 / 10.52 mm |

Noise models (`perception/noise.py`): depth σ(z) = a + b·z² with dropouts, plus colour noise and
lighting changes. "realistic" ≈ a few mm at 1 m; "poor" ≈ 3× worse with 10% missing depth.

## Method comparison under poor noise (30 episodes, cube localization)

| Top-face method | Depth filter | xy error (median) | Height bias |
|---|---|---|---|
| Highest points (95th percentile), first version | none | 7.79 mm | **+6.84 mm** |
| Mode of heights (mean-shift), symmetric band | none | 3.85 mm | −0.95 mm |
| **Mode of heights + 5×5 median filter** (used) | 5×5 | **0.79 mm** | +0.24 mm |

The first version was exact on perfect data and badly biased on noisy data: a method that
is only tested without noise isn't tested.

## What the numbers say

1. **Perception isn't the bottleneck for this task.** Errors of ~1 mm against a 3 cm target and
   a gripper that self-centres on the cube leave a lot of margin. Success is unchanged even with poor noise.
2. **That also means this task can't tell good perception from adequate perception.** A tighter
   target (e.g. placing into a slot), smaller objects, or clutter would. Noted for Stage 7's benchmark.
3. **96% → 98% on the wall scene is not an improvement.** Slightly different estimates change which
   posture the planner picks in one or two borderline episodes; the confidence intervals overlap almost entirely.
4. **Worst cases matter more than medians:** the poor-noise yaw error reaches 8.7° and the target
   error 10.5 mm in single episodes. Harmless here, but closer to failure than the median suggests.

## Known limitations
- Still taken from the simulator: wall geometry (for planning), the cube's pose in the hand after
  grasping, and joint angles (proprioception, which real robots also measure directly).
- One known object with a distinctive colour. Colour thresholds won't generalize to arbitrary objects;
  a learned detector becomes worth it when objects vary (Stages 6-7).
- A single snapshot at the start. If the cube moves (e.g. knocked during the grasp), the robot
  doesn't notice. Closed-loop visual feedback is future work.
