# Stage 4 results: imitation learning (state-based behaviour cloning)

Task: open-table pick-and-place. Every row is the same 100 evaluation episodes (seeds 0-99).
Policies act at 20 Hz through the same action interface; success = cube resting in the target
zone, gripper open and raised, held for 0.5 s, within 25 s. Demonstrations use seeds 10000+,
DAgger rollouts 20000+. Network: 3×256 MLP, 40 epochs, one training seed per row.

"Disturbed": 1 s into the episode, while the gripper is on its way, the cube is moved 5-10 cm.

Reproduce: `scripts/collect_demos.py`, `scripts/train_bc.py`, `scripts/evaluate_bc.py` (see their docstrings).

## Experts

| Expert | Nominal | Disturbed | Median time |
|---|---|---|---|
| Scripted (Stage 2, clock-driven) | 100% (96.3-100) | **0%** (0-3.7) | 11.0 s |
| Reactive (state feedback, no memory) | 100% (96.3-100) | 100% (96.3-100) | 6.8 s |

## Which expert you clone matters more than anything else (500 demonstrations each)

| Demonstrations from | Cloned policy, nominal | Main failures |
|---|---|---|
| Scripted, clock-driven expert | **18%** (11.7-26.7) | stalls, never lifts, wanders off |
| Reactive expert with hard thresholds | 87% (79.0-92.2) | stuck at rule boundaries while placing |
| Reactive expert with continuous "funnel" rules | **100%** (96.3-100) | none |

Same network, same amount of data, same training.

## How many demonstrations (funnel expert)

| Demonstrations | Nominal | Disturbed |
|---|---|---|
| 20 | 7% (3.4-13.7) | not run |
| 100 | 71% (61.5-79.0) | 47% (37.5-56.7) |
| 500 | 100% (96.3-100) | 68% (58.3-76.3) |

## DAgger (3 iterations × 100 learner rollouts, labelled by the reactive expert)

| Starting point | Variant | Nominal | Disturbed |
|---|---|---|---|
| 500 demos | no DAgger | 100% (96.3-100) | 68% (58.3-76.3) |
| 500 demos | DAgger | 97% (91.5-99.0) | **99%** (94.6-99.8) |
| 500 demos | DAgger, repeated stuck states kept | 98% (93.0-99.4) | 96% (90.2-98.4) |
| 100 demos | no DAgger | 71% (61.5-79.0) | 47% (37.5-56.7) |
| 100 demos | DAgger | **39%** (30.0-48.8) | 40% (30.9-49.8) |
| 100 demos | DAgger, repeated stuck states kept | 25% (17.5-34.3) | 32% (23.7-41.7) |

Shipped model: `outputs/models/bc_dagger.pt` (500 demos + DAgger).

## What the numbers say

1. **A policy can only clone an expert that is a function of what the policy sees.** The
   clock-driven expert acts on elapsed time, which the network cannot observe; cloning it gave 18%
   with low validation loss. The state-feedback expert gave 100% with identical training.
2. **Smooth experts clone better than sharp ones** (87% → 100%): hard thresholds in the expert's
   rules are cliffs the network blurs.
3. **On the standard test DAgger looks useless; on the disturbance test it is worth 31 points**
   (68% → 99%). Nominal evaluation only visits states near the demonstrations. The difference
   between 100%, 98% and 97% nominal is within noise.
4. **DAgger made a weak policy worse** (71% → 39%). This contradicts the textbook expectation and
   is only partly explained (below).
5. **The learned policy adapts where its original teacher could not.** No demonstration contains a
   moving cube; reacting to one follows from learning a state → action mapping.
6. **The learned policy does not beat its teacher.** The reactive expert is at 100% in both
   conditions; cloning adds no capability here. Its value is as the foundation for Stage 5
   (RL fine-tuning) and M15 (the same pipeline from images, where no hand-written expert sees pixels).

## Open question: why did DAgger hurt from 100 demonstrations?

**Update:** resolved in `docs/results_m15.md`. Under-training on the aggregated data was the main cause;
with 160 epochs per round DAgger gives 70% nominal and 67% disturbed (plain cloning: 71% / 47%).
The account below is kept as written at the time.

Established:
- Not caused by repeated stuck states alone. Removing them helped a little (25% → 39%) but the
  policy still degraded; that hypothesis was tested and rejected as the main cause.
- The degraded policy fits its training data (low training loss) but not held-out episodes
  (validation loss 15-20× higher).
- Its typical failure is freezing with the gripper open 1-2 cm above the cube. In that region the
  demonstrations always say "down"; the DAgger data also contains "up 25 mm" labels, which are
  correct for the look-alike state after placing (gripper open above the cube at the target).

Not established: that this look-alike is the cause. If it were, freezing should concentrate where
the cube starts near the target; the observed trend (47% of episodes at 12-20 cm separation vs 32%
at 30+ cm) points that way but is within chance for these group sizes.

A decisive test: retrain with an explicit "cube is at the target" input (or the cube-target
distance) and see whether DAgger from 100 demonstrations then improves; and repeat everything
over several training seeds, since each row here is one seed.

## Limitations
- State-based: cube and target poses come from the simulator, not the camera. Image-based
  policies (M15, ACT / Diffusion Policy via LeRobot) need a GPU and are the next step.
- One training seed per configuration; differences of a few points between rows are not meaningful.
- Open-table scene only; the wall scene has not been cloned.
- The disturbance is a single teleport of the cube, one kind of perturbation.
- Menagerie's default (weak) gripper is still in use; the pending stiffness decision is unchanged.
