# Closing the camera loop: learned policies on camera input

**Before:** the learned policies (behaviour cloning, DAgger, residual RL) read the cube's
position from the simulator at every step, which no real robot can do. Only the open-loop
scripted and planned policies used the camera (Stage 3), from one snapshot at the start.

**Now:** every policy can read the cube and target from images, re-estimated at every 50 ms
policy step (`perception/tracking.py`). Robot joint angles and finger opening still come from
the simulator, as encoders would provide them on a real robot. Success is always judged on the
true state.

Reproduce: `python scripts/benchmark.py --camera --n 50` (about 4 h on 2 CPU cores: every
policy step renders a 640×480 RGB-D image), then `python scripts/benchmark_report.py`.
Per-episode results with tracking statistics: `outputs/benchmark/camera.json`.

## How the cube is tracked

Per policy step, in order:

1. **Wrist camera** (on the hand, looking down between the fingers), when it sees the whole cube.
2. **Front camera** (fixed, sees the table), when the wrist camera does not.
3. **Neither:** if the finger opening says the gripper is holding the cube, the cube moves with
   the hand at the offset last seen; otherwise the last estimate is kept.

The target is located once, by the front camera, at the start.

The order comes from a measurement made before building anything (6 episodes, 774 steps):

| Camera | Cube free: median error | Cube in the hand: median error |
|---|---|---|
| Front | 0.5 mm | 10.3 mm (the hand hides the cube's top face) |
| Wrist | 0.4 mm | 0.6 mm |

**A bug found by the first full run.** "Sees the whole cube" was not in the first version.
When the cube started near the edge of the wrist camera's view, the camera saw part of it and
the estimate was off by up to 20 mm. The open-loop scripted policy reads the cube only once,
so it missed (seed 14 of the outside-area condition). The tracker now projects the estimated
cube back into the image and distrusts any camera whose view cuts it off. The first-frame error
dropped to at most 1.1 mm over 100 starts. The run was stopped halfway and restarted, so every
number below comes from the fixed version.

## Results: same policies, same seeds, simulator state vs cameras

50 episodes per cell (seeds 0-49), compared episode by episode with the simulator-state
benchmark on the same seeds. Each cell: successes with simulator state → with cameras.

| Policy | Nominal | Cube moved | Outside area | 3 cm cube | 5 cm cube | Latency 100 ms |
|---|---|---|---|---|---|---|
| Scripted (open-loop) | 50 → 50 (+0) | 0 → 0 (+0) | 50 → 50 (+0) | 50 → 50 (+0) | 50 → 50 (+0) | 50 → 50 (+0) |
| Reactive (feedback) | 50 → 50 (+0) | 50 → 50 (+0) | 50 → 49 (-1) | 50 → 50 (+0) | 50 → 50 (+0) | 50 → 50 (+0) |
| Clone, 100 demos | 33 → 32 (-1) | 23 → 20 (-3) | 27 → 22 (-5) | 12 → 8 (-4) | 18 → 14 (-4) | 37 → 36 (-1) |
| Clone, 500 demos | 50 → 50 (+0) | 33 → 32 (-1) | 47 → 46 (-1) | 44 → 45 (+1) | 39 → 39 (+0) | 50 → 50 (+0) |
| Clone + DAgger | 48 → 48 (+0) | 49 → 49 (+0) | 45 → 44 (-1) | 43 → 45 (+2) | 4 → 4 (+0) | 48 → 48 (+0) |
| 100-demo clone + RL | 46 → 47 (+1) | 38 → 37 (-1) | 41 → 35 (-6) | 13 → 9 (-4) | 40 → 33 (-7) | 47 → 45 (-2) |

Totals over the 300 paired episodes of each policy, with an exact McNemar test (are the
episodes lost and gained balanced?):

| Policy | Simulator state | Cameras | Lost | Gained | p |
|---|---|---|---|---|---|
| Scripted (open-loop) | 250 | 250 | 0 | 0 | 1.00 |
| Reactive (feedback) | 300 | 299 | 1 | 0 | 1.00 |
| Clone, 500 demos | 263 | 262 | 9 | 8 | 1.00 |
| Clone + DAgger | 237 | 238 | 2 | 3 | 1.00 |
| **Clone, 100 demos** | 150 | 132 | 23 | 5 | **0.001** |
| **100-demo clone + RL** | 225 | 206 | 29 | 10 | **0.003** |

## What this shows

**1. For the four strong policies, the camera loop costs nothing measurable.** The net change
is at most one episode in 300 for each, and no single condition moves by more than two
episodes in 50. This is what the Stage 7 sensing sweeps
predicted: these policies tolerated 2 mm of random sensing error, and the tracker's typical
error is 0.5 mm.

**2. The two weakest policies lose significantly: 18 and 19 episodes of 300.** They are the
policies that stall or hesitate at the grasp. That is exactly where tracking is worst (point 3).
The RL policy loses most on the 5 cm cube (40 → 33) and outside the training area (41 → 35);
the 100-demo clone loses across all six conditions. The episodes lost and gained are not
balanced (p < 0.01), so this is not noise.

**3. Tracking is accurate except when the open gripper hovers around the cube.**

| Condition | Median error (mm) | Worst 5% of episodes: their 95th-percentile error (mm) | Wrist camera | Front camera | Moved with hand | Kept last | Target error, median (mm) |
|---|---|---|---|---|---|---|---|
| Nominal | 0.51 | 15.26 | 99.3% | 0.7% | 0.0% | 0.0% | 0.88 |
| Cube moved | 0.54 | 26.34 | 81.2% | 18.8% | 0.0% | 0.0% | 0.88 |
| Outside area | 0.56 | 20.65 | 96.2% | 3.8% | 0.0% | 0.0% | 0.96 |
| 3 cm cube | 0.58 | 10.83 | 97.0% | 3.0% | 0.0% | 0.0% | 0.88 |
| 5 cm cube | 0.53 | 18.15 | 99.7% | 0.3% | 0.0% | 0.0% | 0.88 |
| Latency 100 ms | 0.52 | 14.71 | 99.5% | 0.5% | 0.0% | 0.0% | 0.88 |

The typical error is about 0.5 mm in every condition. The large values in the second column
come from one situation, logged step by step: the gripper open around the cube at grasp
height, not closing. The fingers then hide part of the cube's top face from the wrist camera,
and the estimate drifts 5-10 mm sideways. Successful episodes have a median per-episode
95th-percentile error of 0.8 mm; failed episodes 2.4 mm. A policy that grasps promptly never
gives the tracker this view. One that stalls does, and the drifting estimate then makes it
harder to recover. Whether the drift causes these failures or only accompanies them is not
separated here.

**4. One remaining failure of the reactive policy is not a tracking failure.** Seed 34,
outside-area condition: tracking stayed within 1.1 mm the whole episode, yet the arm grasped
and never lifted. The camera's height estimate sits consistently about 0.5 mm low, so the
gripper closed slightly deeper. At this spot, on the near edge of the workspace, the grasp
jammed. Observed in one episode; the mechanism is not proven.

**5. The "held" fallback was never needed.** One camera or the other saw the cube in every
step of all 1,800 episodes. The fallback is tested (`tests/test_tracking.py`), but this
benchmark never exercised it.

## Limits

- 50 episodes per cell, one training seed per policy. The paired comparison makes small
  differences detectable, but the per-cell numbers have wide intervals.
- Simulated cameras: perfect calibration, no motion blur, no lens distortion, no exposure
  changes, rendered without shadows or reflections. The noise models of Stage 3 were not
  applied here. A real camera adds all of these.
- Colour segmentation: a red cube and a green target on a plain table. Clutter, similar
  colours and occlusion by other objects are not tested.
- Joint angles and finger opening still come from the simulator (as encoders would).
- Stage 6's language and multi-object scene are still state-based.
- The image policy (Stage 4, M15) is not part of this: it reads pixels directly.
