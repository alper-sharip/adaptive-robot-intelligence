# M15 results: learning from images (CPU-scale run)

Task and evaluation are those of Stage 4: open-table pick-and-place, the same 100 evaluation
episodes (seeds 0-99), 20 Hz actions. The student policy receives two 64×64 camera images
(front, wrist) and the gripper's own pose. It never receives the cube or target position.

**Scale of this run:** everything here was produced on 2 CPU cores, with no GPU: 200
demonstrations, 18 epochs, a 0.2 M-parameter network, images rendered without shadows or
reflections. It verifies the pipeline and gives a baseline. It is not what the method can do
at full scale; `notebooks/m15_vision_policy_colab.ipynb` runs the same scripts on a GPU.

Reproduce: `python scripts/train_vision.py --demos 200 --epochs 18 --name vision_cpu`, then
`python scripts/evaluate_bc.py [--disturb] outputs/models/vision_cpu.pt`.

## Image-based student vs state-based policies

| Policy | Input | Demos | Nominal | Cube moved mid-episode |
|---|---|---|---|---|
| Reactive expert (teacher) | simulator state | - | 100% (96.3-100) | 100% (96.3-100) |
| Stage 4 clone | simulator state | 100 | 71% (61.5-79.0) | 47% (37.5-56.7) |
| Stage 4 clone | simulator state | 500 | 100% (96.3-100) | 68% (58.3-76.3) |
| **Image student** (`vision_cpu.pt`) | 2 cameras + gripper pose | 200 | **43%** (33.7-52.8) | **4%** (1.6-9.8) |

Image student, failures on the nominal set (57): placed in the wrong spot or dropped 45,
did not retreat 6, never lifted 5, still holding 1. Median placement error of successes: 11.1 mm
(state-based policies: about 2 mm).

## What the numbers say

1. **The student learned to grasp from pixels.** It lifted the cube in 95 of 100 episodes, on
   cubes at random positions and angles it had not seen.
2. **Placing is what fails.** While carrying, the wrist camera looks at the cube, not the target;
   the target must be located in the front image, where it is ~6 pixels wide and one pixel is
   about 1 cm. Placement error is 5× the state-based policies'. Candidate fixes, untested:
   higher resolution, a wider or differently aimed wrist camera, more data.
3. **It does not adapt.** With the cube moved mid-episode it succeeds 4 times in 100 (85 "never
   lifted"). The state-based clones kept 47-68% without DAgger. Plain cloning from images at this
   scale learned the nominal sequence, not a robust "go to where the cube is now". This is the
   gap DAgger is meant to close, and DAgger is what failed here (below).
4. **Images cost precision.** Validation error at the end of training was 0.0175 (normalized),
   against 0.0016 for the 500-demo state-based policy.

## DAgger on the image student: a negative result at this scale

| Attempt | Fine-tuning per DAgger round | Student success on DAgger rollouts (60 episodes, seeds 20000+) |
|---|---|---|
| 1 | 3 epochs | 23/60 before → 1/60 after one round |
| 2 | 8 epochs | 24/60 before → 2/60 after one round |

After adding the DAgger frames, training error was 5-10× higher than before (0.03-0.05 vs 0.006)
and validation error 20-40× higher. The network did not fit the aggregated data in the epochs
given, and what it had learned was damaged. Whether much longer training, a larger network or
more base demonstrations fixes this is not established here; the state-based experiments below
suggest training budget is at least part of it. The shipped model is the one without DAgger.

## Stage 4's open question, revisited (state-based)

"Why did DAgger make a weak policy worse?" Three hypotheses were tested by intervention:

| Hypothesis | Intervention | Result (100 demos + DAgger, nominal) | Verdict |
|---|---|---|---|
| Repeated stuck states swamp the data | drop repeats | 25% → 39% | minor factor |
| States on the expert's decision cliffs poison the labels | drop cliff states (4.0% of weak-learner DAgger states vs 0.4% of demos) | 35% | rejected as cause |
| Fixed training budget on a larger, more varied dataset → less precise fit | retrain 160 epochs instead of 40 | **70%** | supported |

| 100 demonstrations | Nominal | Cube moved |
|---|---|---|
| Plain cloning | 71% (61.5-79.0) | 47% (37.5-56.7) |
| + DAgger, 40 epochs per round | 39% (30.0-48.8) | 40% (30.9-49.8) |
| + DAgger, 160 epochs per round | 70% (60.4-78.1) | **67%** (57.3-75.4) |

A direct look at frozen states showed the mechanism: the gripper sits 3-4 mm off-centre just
above the expert's grasp trigger, the expert asks for a millimetre-scale correction, and a policy
whose error is of that size outputs nothing. Training error tracks it: 0.0007 (500 demos),
0.0053 (100 demos), 0.0079 (after DAgger, 40 epochs), 0.0011 (same data, 200 epochs).

So: DAgger did not hurt; under-training on DAgger's data did. With adequate training it is
neutral on the nominal test and worth 20 points under disturbance. It did not lift nominal
success above plain cloning at this data size. One training seed per row.

## ACT via LeRobot: status
`scripts/lerobot_act.py` converts the demonstrations to a LeRobotDataset, trains LeRobot's ACT
(18.7 M parameters) and rolls it out in this simulator. Verified on CPU with lerobot 0.7.0:
conversion of 4 episodes, 6 training steps, save, reload, one rollout. **No trained ACT policy
exists yet and no success rate is claimed.** Full training needs a GPU.

## Limitations
- CPU-scale: small network, low resolution, 200 demonstrations, one training seed.
- No shadows or reflections in the student's images; robustness to them is untested.
- The teacher still uses simulator state; that is the point of teacher-student training, and it
  restricts this method to simulation (or to a real setup with extra instrumentation).
- Open-table scene only.
