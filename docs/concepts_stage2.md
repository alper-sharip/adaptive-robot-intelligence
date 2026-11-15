# Concept notes: Stage 2

## 1. Building scenes in code: `envs/pick_place.py`
`mujoco.MjSpec` loads the Menagerie scene and adds bodies programmatically (cube with a
free joint, target site). A **free joint** gives a body 6 DoF: `qpos` gets 7 numbers
(xyz + quaternion), `qvel` gets 6. Sites are markers: they never collide.

## 2. Randomization and seeds: `PickPlaceEnv.sample_episode`
All randomness flows from one `np.random.default_rng(seed)`. Seed 17 is the same episode
everywhere. Evaluation uses seeds 0-99; training data will use 10000+, so a learned policy is
never tested on episodes it trained on (the robotics version of a train/test split).

## 3. Cartesian control: `control/cartesian.py`
Three rates: trajectory → IK at 100 Hz → servos and physics at 500 Hz. IK is warm-started
from the previous *command*, so it converges in 1-3 iterations and the arm can't jump to a
different IK solution mid-motion. Later, learned policies output TCP poses into this same
controller; this is the "action space" decision from the project evaluation.

## 4. Minimum-jerk trajectories: `planning/trajectory.py`
`s(τ) = 10τ³ − 15τ⁴ + 6τ⁵`: zero velocity and acceleration at both ends. Peak speed is
1.875× the average, which is how `duration_for` turns a speed limit into a duration.
Rotations are interpolated with the exp/log maps: `R(s) = R0 · exp(s · log(R0ᵀR1))`,
a constant-speed turn about one fixed axis (the matrix form of quaternion slerp).

## 5. Grasping physics
- Holding requires friction ≥ weight: `2 · μ · F_normal ≥ m·g` (two fingers). Add the lift
  acceleration (`m·(g + a)`) and the margin shrinks further.
- A cube has 4-fold symmetry, so any grasp yaw ≡ cube yaw (mod 90°) works; `grasp_yaw`
  picks the one in [−45°, 45°] to keep wrist rotations small.
- Simulated contact is *soft*: objects can creep and interpenetrate slightly. The friction
  cone model and solver settings (impratio) change how much.

## 6. Evaluation that means something: `scripts/evaluate_scripted.py`
- **Confidence intervals:** 100/100 successes is consistent with a true rate of 96.3%. The
  Wilson interval is printed for every run. Two policies whose intervals overlap heavily
  haven't been shown to differ.
- **Failure attribution:** checks at phase boundaries turn "failed" into "grasp_failed",
  "dropped_in_transport", "placement_miss". This tells you what to fix next.
- **Stress tests:** a baseline at 100% tells you nothing until you find where it breaks.

## Check your understanding
1. With μ = 0.7 and the default gripper, predict success or failure before running it. Then run
   `--friction 0.7`.
2. Why is the warm start taken from the previous *command* rather than the *measured* joints?
3. In `grasp_yaw`, what would go wrong if we always used the cube's raw yaw?
4. Placement error grows with speed even with no slip (results rows 8-9). Propose a test that
   distinguishes "arm still moving at release" from "IK tracking lag".
