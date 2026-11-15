# Concept notes: Stage 4 imitation learning

## 1. Behaviour cloning is supervised learning: `learning/bc.py`
Collect (observation, expert action) pairs, fit a network to predict the action. The loss says
nothing about cubes or targets; the network is only rewarded for copying. Motion outputs use
mean-squared error; the open/close gripper output is a classification (binary cross-entropy).

## 2. Observation and action spaces decide what is learnable: `learning/spaces.py`
- Everything the policy can know must be in the observation. If two situations that need
  different actions produce the same observation, no network can tell them apart.
- Encodings matter: angles as sin/cos (no jump at ±180°); the cube's yaw as sin 4θ, cos 4θ so
  that a cube turned by 90° gives the identical input.
- Actions are small moves relative to the *measured* gripper pose, at 20 Hz. The controller
  from Stages 1-2 (IK + joint servos) executes them: classical control underneath, learning on top.

## 3. Low loss is not success
The first policy predicted the expert's actions well on held-out data and succeeded 20% of the
time. Validation loss measures one-step prediction on the EXPERT's states. A rollout chains
hundreds of predictions on the LEARNER's states. Only rollouts measure the task.

## 4. The expert must be a function of what the learner sees
The clock-driven Stage 2 expert does different things in the same state depending on elapsed
time (slow at the start of a move, fast in the middle, pauses of fixed length). A memoryless
network trained on that sees conflicting labels and learns their average, often "barely move".
Two ways out:
- make the expert a function of the state (`skills/reactive_pick_place.py`), or
- give the learner memory: observation history or action chunks (ACT, Diffusion Policy: M15).

## 5. Compounding errors and distribution shift
Each small prediction error moves the robot to a state slightly unlike the demonstrations,
where the next prediction is a little worse, and so on. The expert's data covers only the
narrow tube of states the expert visits. Remedies:
- **DAgger:** run the learner, label the states *it* reaches with the expert's action, retrain.
- **A forgiving expert:** continuous rules (funnels) so that near-miss states have sensible,
  nearby labels. Hard thresholds in the expert become cliffs the network cannot fit.

## 6. Train / validation / test, robotics version
- Split by episode, never by sample: neighbouring samples are near-duplicates.
- Demonstration seeds 10000+, DAgger seeds 20000+, evaluation seeds 0-99: disjoint.
- Normalization statistics come from the training split only.

## 7. Evaluation interface details that changed results
- The expert must act through the same 20 Hz interface as the learner, or its recorded actions
  do not reproduce its own behaviour.
- Commands are interpolated between decisions; jumping to each new target jolted the arm enough
  for the weak gripper to drop the cube.

## Check your understanding
1. Why did validation loss go UP after DAgger iterations with the threshold-based expert, even
   though more data was added?
2. The reactive expert has no clock. How does it "know" that the grasp has closed, or that the
   task is finished? Which observation entries carry that information?
3. Give one situation where even a perfect state-feedback expert would produce conflicting
   labels for a memoryless learner. (Hint: think about what the 20 numbers leave out.)
4. The cloned policy handles a moved cube although no demonstration ever showed a cube moving.
   Why is that possible here, and why would it not be with the clock-driven expert's data?
