# Stage 2 results: scripted pick-and-place baseline

Task: pick a 4 cm, 100 g cube from a random position and yaw, place it in a random
target zone (success = cube on the table, centre within 3 cm of the target).
Evaluation set: seeds 0-99, identical for every row. Cube pose comes from simulator
ground truth (Stage 3 replaces it with vision).

Reproduce any row with `python scripts/evaluate_scripted.py <flags>`.

| # | Configuration | Flags | Success (95% CI) | Failures | Placement error (median) | Slip in grip (median) | Episode time |
|---|---|---|---|---|---|---|---|
| 1 | **Default**: μ=1.0, Menagerie gripper, 0.25 m/s, elliptic cone | *(none)* | **100%** (96.3-100) | none | 1.9 mm | 0.4 mm | 11.3 s |
| 2 | MuJoCo default friction cone (pyramidal) | `--pyramidal` | 100% (96.3-100) | none | 1.1 mm | 3.0 mm | 11.3 s |
| 3 | Lower friction μ=0.5 | `--friction 0.5` | **0%** (0-3.7) | grasp_failed ×100 | - | - | 11.3 s |
| 4 | Low friction μ=0.3 | `--friction 0.3` | **0%** (0-3.7) | grasp_failed ×100 | - | - | 11.3 s |
| 5 | μ=0.3, gripper stiffness ×10 | `--friction 0.3 --gripper-stiffness 10` | 100% (96.3-100) | none | 2.0 mm | 0.2 mm | 11.3 s |
| 6 | Faster: 0.6 m/s | `--max-speed 0.6` | 96% (90.2-98.4) | dropped ×4 | 6.0 mm | 2.9 mm | 7.3 s |
| 7 | Fastest: 1.0 m/s | `--max-speed 1.0` | 89% (81.4-93.7) | dropped ×8, grasp ×2, placement ×1 | 8.8 mm | 5.9 mm | 6.6 s |
| 8 | 0.6 m/s, gripper ×10 | `--max-speed 0.6 --gripper-stiffness 10` | 100% (96.3-100) | none | 4.6 mm | 0.8 mm | 7.3 s |
| 9 | 1.0 m/s, gripper ×10 | `--max-speed 1.0 --gripper-stiffness 10` | 100% (96.3-100) | none | 6.2 mm | 1.4 mm | 6.6 s |
| 9a | *same as 9, before the damping fix* | - | *7%* (3.4-13.7) | *grasp_failed ×93* | - | - | - |

## What the numbers say

1. **The baseline is solid under nominal conditions**, but that alone says little: 100% on easy
   conditions only means the conditions are easy. Rows 3-9 show where it breaks.
2. **Grip force is the limiting factor.** Menagerie's gripper presses each finger with only
   ~1.0 N (measured). Friction capacity is 2 · μ · 1 N against a 0.98 N cube:
   μ = 1.0 gives 2× margin; μ = 0.5 gives none, so 0/100. The real Panda grips with ~70 N.
3. **Speed trades success and accuracy for time.** Going from 0.25 to 1.0 m/s saves 42% of the
   episode time but costs 11 points of success (weak grip) and roughly 3-4× the placement error.
   With a stronger grip the success cost disappears, but the placement error does not.
4. **Damping ratios matter when you change stiffness** (row 9a vs 9): see devlog.md.

## Open questions (good exercises)
- Why does placement error grow with speed even when nothing slips (rows 8-9)? Hypotheses to
  test: the arm still moving at release; IK tracking lag; the cube swinging in the grip.
- Is 100/100 at defaults the true rate? The 95% CI says anything from 96.3% upward is consistent.
  How many episodes would you need to tell 98% from 99.5%?

## Decision pending
Default gripper stiffness is the unmodified Menagerie value (×1). Rows 3-9 argue for ×10
(a still-modest ~10 N per finger) as the default for later stages, especially Stage 7's
varied objects. This is a modelling choice to make deliberately, not silently.
