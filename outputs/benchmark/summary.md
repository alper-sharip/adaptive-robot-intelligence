## Main matrix: success rate (%), 100 episodes per cell

| Policy | Nominal | Cube moved | Latency 100 ms | Sensing jitter | Sensing offset | Outside area | 3 cm cube | 5 cm cube | Heavy cube | Friction 0.7 | Friction 0.5 | Stiff gripper | Friction 0.5, stiff | Mean | Worst |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Scripted (open-loop) | 100 | 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 0 | 100 | 100 | 84.6 | 0 |
| Reactive (feedback) | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 0 | 100 | 100 | 1 | 100 | 100 | 84.5 | 0 |
| Clone, 100 demos | 66 | 42 | 39 | 85 | 70 | 54 | 20 | 41 | 73 | 72 | 1 | 67 | 67 | 53.6 | 1 |
| Clone, 500 demos | 100 | 63 | 74 | 100 | 100 | 92 | 90 | 78 | 99 | 100 | 3 | 100 | 100 | 84.5 | 3 |
| Clone + DAgger | 97 | 99 | 53 | 100 | 98 | 89 | 88 | 8 | 96 | 98 | 0 | 97 | 97 | 78.5 | 0 |
| 100-demo clone + RL | 93 | 73 | 49 | 96 | 88 | 78 | 34 | 71 | 86 | 93 | 0 | 92 | 83 | 72.0 | 0 |

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

| Action latency (ms) | 0 | 50 | 100 | 150 | 200 |
|---|---|---|---|---|---|
| Scripted (open-loop) | 100 | 100 | 100 | 100 | 11 |
| Reactive (feedback) | 100 | 100 | 100 | 100 | 44 |
| Clone, 100 demos | 66 | 71 | 39 | 10 | 0 |
| Clone, 500 demos | 100 | 100 | 74 | 27 | 16 |
| Clone + DAgger | 97 | 99 | 53 | 37 | 34 |
| 100-demo clone + RL | 93 | 81 | 49 | 23 | 6 |

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
| Reactive (feedback) | 201 of 1300 | 193 | 5 | 2 | 1 |
| Clone, 100 demos | 603 of 1300 | 368 | 164 | 67 | 4 |
| Clone, 500 demos | 201 of 1300 | 149 | 18 | 25 | 9 |
| Clone + DAgger | 280 of 1300 | 155 | 92 | 24 | 9 |
| 100-demo clone + RL | 364 of 1300 | 191 | 21 | 149 | 3 |

## Image policy (two 64x64 cameras), success rate (%)

| Nominal | Cube moved | Outside area |
|---|---|---|
| 42 (33-52) | 4 (2-10) | 7 (3-14) |

## Latency, two ways: success rate (%)

| Policy | No delay | 100 ms, step relative to the pose at application | 100 ms, original pose target kept | 200 ms, relative | 200 ms, target kept |
|---|---|---|---|---|---|
| Scripted (open-loop) | 100 | 100 | 100 | 11 | 100 |
| Reactive (feedback) | 100 | 100 | 100 | 44 | 100 |
| Clone, 100 demos | 66 | 39 | 78 | 0 | 83 |
| Clone, 500 demos | 100 | 74 | 100 | 16 | 99 |
| Clone + DAgger | 97 | 53 | 97 | 34 | 97 |
| 100-demo clone + RL | 93 | 49 | 94 | 6 | 93 |

## Footnote measurements

| Cell | Success |
|---|---|
| reactive#size|large | 100/100 |
| reactive#size|small | 100/100 |
| reactive|heavy@0.2 | 0/100 |
| outputs/models/bc_dagger.pt|heavy@0.2 | 0/100 |
