## Main matrix: success rate (%), 100 episodes per cell

| Policy | Nominal | Cube moved | Latency 100 ms | Sensing jitter | Sensing offset | Outside area | 3 cm cube | 5 cm cube | Heavy cube | Friction 0.7 | Friction 0.5 | Stiff gripper | Friction 0.5, stiff | Mean | Worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Scripted (open-loop) | 100 | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 84.6 | 0 |
| Reactive (feedback) | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 100 | 100 | 1 | 100 | 100 | 92.2 | 1 |
| Clone, 100 demos | 66 | 42 | 78 | 85 | 70 | 54 | 20 | 41 | 73 | 72 | 1 | 67 | 67 | 56.6 | 1 |
| Clone, 500 demos | 100 | 63 | 100 | 100 | 100 | 92 | 90 | 78 | 99 | 100 | 3 | 100 | 100 | 86.5 | 3 |
| Clone + DAgger | 97 | 99 | 97 | 100 | 98 | 89 | 88 | 8 | 96 | 98 | 0 | 97 | 97 | 81.8 | 0 |
| 100-demo clone + RL | 93 | 73 | 95 | 96 | 88 | 78 | 34 | 71 | 86 | 93 | 0 | 92 | 83 | 75.5 | 0 |

## Conditions

| Condition | What differs from training |
|---|---|
| Nominal | the training conditions |
| Cube moved | cube moved 5-10 cm, 1 s into the episode |
| Latency 100 ms | every action takes effect 100 ms late |
| Sensing jitter | cube position sensed with 2 mm random error at every step |
| Sensing offset | cube position sensed 5 mm off, constant per episode |
| Outside area | cube starts up to 5 cm outside the training area |
| 3 cm cube | 3 cm cube instead of 4 cm |
| 5 cm cube | 5 cm cube instead of 4 cm |
| Heavy cube | cube mass 0.15 kg instead of 0.1 |
| Friction 0.7 | cube friction 0.7 instead of 1.0 |
| Friction 0.5 | cube friction 0.5 instead of 1.0 |
| Stiff gripper | gripper servo 10x stiffer |
| Friction 0.5, stiff | cube friction 0.5 AND gripper servo 10x stiffer |

## Severity sweeps: success rate (%)

| Sensing jitter (mm) | 0 | 1 | 2 | 5 | 10 |
|---|---|---|---|---|---|
| Scripted (open-loop) | 100 | 100 | 100 | 100 | 88 |
| Reactive (feedback) | 100 | 100 | 100 | 1 | 0 |
| Clone, 100 demos | 66 | 79 | 85 | 89 | 83 |
| Clone, 500 demos | 100 | 100 | 100 | 100 | 97 |
| Clone + DAgger | 97 | 98 | 100 | 100 | 98 |
| 100-demo clone + RL | 93 | 94 | 96 | 91 | 73 |

| Sensing offset (mm) | 0 | 2 | 5 | 10 | 20 |
|---|---|---|---|---|---|
| Scripted (open-loop) | 100 | 100 | 100 | 100 | 68 |
| Reactive (feedback) | 100 | 100 | 100 | 99 | 3 |
| Clone, 100 demos | 66 | 66 | 70 | 65 | 17 |
| Clone, 500 demos | 100 | 99 | 100 | 92 | 19 |
| Clone + DAgger | 97 | 98 | 98 | 88 | 20 |
| 100-demo clone + RL | 93 | 96 | 88 | 81 | 38 |

| Action latency (ms) | 0 | 50 | 100 | 150 | 200 | 300 | 500 |
|---|---|---|---|---|---|---|---|
| Scripted (open-loop) | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| Reactive (feedback) | 100 | 100 | 100 | 100 | 100 | 45 | 0 |
| Clone, 100 demos | 66 | 77 | 78 | 82 | 83 | 28 | 0 |
| Clone, 500 demos | 100 | 100 | 100 | 100 | 99 | 46 | 0 |
| Clone + DAgger | 97 | 97 | 97 | 97 | 97 | 44 | 0 |
| 100-demo clone + RL | 93 | 94 | 95 | 93 | 93 | 37 | 0 |

## Placement precision in the training conditions (successful episodes)

| Policy | Median error (mm) | 95th percentile (mm) | Within 10 mm (% of all episodes) | Within 5 mm (% of all episodes) |
|---|---|---|---|---|
| Scripted (open-loop) | 1.7 | 2.9 | 100 | 100 |
| Reactive (feedback) | 1.5 | 2.3 | 100 | 100 |
| Clone, 100 demos | 4.0 | 10.7 | 62 | 39 |
| Clone, 500 demos | 1.4 | 3.9 | 99 | 97 |
| Clone + DAgger | 1.6 | 2.7 | 97 | 96 |
| 100-demo clone + RL | 11.9 | 23.7 | 35 | 10 |

## Failure types, summed over the 13 conditions

| Policy | Episodes failed | never lifted | still holding / off table | dropped or misplaced | placed, did not retreat |
|---|---|---|---|---|---|
| Scripted (open-loop) | 200 of 1300 | 200 | 0 | 0 | 0 |
| Reactive (feedback) | 101 of 1300 | 93 | 5 | 2 | 1 |
| Clone, 100 demos | 564 of 1300 | 325 | 169 | 65 | 5 |
| Clone, 500 demos | 175 of 1300 | 126 | 16 | 24 | 9 |
| Clone + DAgger | 236 of 1300 | 112 | 91 | 24 | 9 |
| 100-demo clone + RL | 318 of 1300 | 175 | 20 | 120 | 3 |

## Image policy (two 64x64 cameras), success rate (%)

| Nominal | Cube moved | Outside area |
|---|---|---|
| 42 (33-52) | 4 (2-10) | 7 (3-14) |

## Latency before and after the fix: success rate (%)

Old: steps relative to the pose when the action is applied. Fixed: relative to the pose the policy observed.

| Policy | No delay | 50 ms old | 50 ms fixed | 100 ms old | 100 ms fixed | 150 ms old | 150 ms fixed | 200 ms old | 200 ms fixed |
|---|---|---|---|---|---|---|---|---|---|
| Scripted (open-loop) | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 11 | 100 |
| Reactive (feedback) | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 44 | 100 |
| Clone, 100 demos | 66 | 71 | 77 | 39 | 78 | 10 | 82 | 0 | 83 |
| Clone, 500 demos | 100 | 100 | 100 | 74 | 100 | 27 | 100 | 16 | 99 |
| Clone + DAgger | 97 | 99 | 97 | 53 | 97 | 37 | 97 | 34 | 97 |
| 100-demo clone + RL | 93 | 81 | 94 | 49 | 95 | 23 | 93 | 6 | 93 |

## Footnote measurements

| Cell | Success |
|---|---|
| reactive#4cm|small | 100/100 |
| reactive#4cm|large | 0/100 |
| reactive|heavy@0.2 | 0/100 |
| outputs/models/bc_dagger.pt|heavy@0.2 | 0/100 |

## Camera loop: success with cube and target from images vs from the simulator

Same policies, same seeds. Each cell: simulator state → cameras (episodes of 50); change in brackets.

| Policy | Nominal | Cube moved | Outside area | 3 cm cube | 5 cm cube | Latency 100 ms |
|---|---|---|---|---|---|---|
| Scripted (open-loop) | 50 → 50 (+0) | 0 → 0 (+0) | 50 → 50 (+0) | 50 → 50 (+0) | 50 → 50 (+0) | 50 → 50 (+0) |
| Reactive (feedback) | 50 → 50 (+0) | 50 → 50 (+0) | 50 → 49 (-1) | 50 → 50 (+0) | 50 → 50 (+0) | 50 → 50 (+0) |
| Clone, 100 demos | 33 → 32 (-1) | 23 → 20 (-3) | 27 → 22 (-5) | 12 → 8 (-4) | 18 → 14 (-4) | 37 → 36 (-1) |
| Clone, 500 demos | 50 → 50 (+0) | 33 → 32 (-1) | 47 → 46 (-1) | 44 → 45 (+1) | 39 → 39 (+0) | 50 → 50 (+0) |
| Clone + DAgger | 48 → 48 (+0) | 49 → 49 (+0) | 45 → 44 (-1) | 43 → 45 (+2) | 4 → 4 (+0) | 48 → 48 (+0) |
| 100-demo clone + RL | 46 → 47 (+1) | 38 → 37 (-1) | 41 → 35 (-6) | 13 → 9 (-4) | 40 → 33 (-7) | 47 → 45 (-2) |

## Tracking accuracy (all camera episodes)

| Condition | Median error (mm) | 95th percentile of per-episode 95th percentile (mm) | Wrist camera | Front camera | Moved with hand | Kept last | Target error, median (mm) |
|---|---|---|---|---|---|---|---|
| Nominal | 0.51 | 15.26 | 99.3% | 0.7% | 0.0% | 0.0% | 0.88 |
| Cube moved | 0.54 | 26.34 | 81.2% | 18.8% | 0.0% | 0.0% | 0.88 |
| Outside area | 0.56 | 20.65 | 96.2% | 3.8% | 0.0% | 0.0% | 0.96 |
| 3 cm cube | 0.58 | 10.83 | 97.0% | 3.0% | 0.0% | 0.0% | 0.88 |
| 5 cm cube | 0.53 | 18.15 | 99.7% | 0.3% | 0.0% | 0.0% | 0.88 |
| Latency 100 ms | 0.52 | 14.71 | 99.5% | 0.5% | 0.0% | 0.0% | 0.88 |
