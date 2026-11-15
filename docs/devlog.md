# Development log

Findings, bugs and their fixes. Interviewers ask "what went wrong?". This file holds the answers.

## 2026-11-14: Foundation (M0–M4)

### Kinematics cross-check
Forward kinematics was implemented from Franka's modified DH table and compared with
MuJoCo's body poses (built from MJCF offsets/quaternions). Agreement: position to ~1e-16 m,
rotation to ~3e-8. The rotation residual is because the MJCF stores quaternions like
0.70710678 to 8 digits. That's a nice example of why tests need tolerances.

### Bug: IK stalled at ~0.5 mm error
**Symptom:** every IK target "failed" after 200 iterations, stuck at 0.5–0.7 mm.
**Diagnosis:** setting `nullspace_gain=0` made IK converge in 4 iterations, so the
null-space term was the cause.
**Root cause:** the null-space projector `N = I - J⁺J` was built with the *damped*
pseudo-inverse. With damping, `J⁺J ≠ I` on the task space, so `N` is not a true null-space
projector: part of the "posture" motion leaked into the TCP and fought the task error.
**Fix:** use the exact pseudo-inverse (`np.linalg.pinv`) for `N`. IK now converges in 4–9 iterations.

### Finding: gravity sag of the position servos
The Menagerie Panda's actuators are PD servos with no gravity compensation. Holding a pose,
the joints sag until `kp · error = gravity torque`. Measured: joint 2 sits 8.7 mrad off target.
At the TCP that becomes 3–9 mm. Adding `data.qfrc_applied = data.qfrc_bias` (gravity +
Coriolis feedforward) reduced TCP error to ~0.06 mm.
**Caveat:** `qfrc_applied` bypasses the actuator torque limits, so this feedforward is
"free" torque. That's acceptable for now, but it should be revisited when we move to
torque/operational-space control.

### Finding (corrected): what sets the step-response speed
*First version of this entry was wrong.* I claimed rise time was set by the torque limits,
because the torque plot shows flat tops at 87 / 12 Nm. Testing that claim:

| kp scale | kd scale | rise time (joint 1) |
|---|---|---|
| 1 | 1 | 0.212 s |
| 0.5 | 0.5 | 0.204 s |
| 1 | 0.5 | 0.108 s |
| 1 | 2 | 0.434 s |

Rise time tracks **kd/kp**, not kp alone and not the torque limit. The Menagerie servos are
heavily overdamped (kd/kp = 0.1 s on every joint), so the damping term dominates the inertia:
`kd·q̇ ≈ kp·(target − q)`, a first-order system with time constant kd/kp = 0.1 s, whose
10–90% rise time is 2.2 × 0.1 = 0.22 s. Saturation only lasts the first ~30 ms.
Lesson: test the explanation, not just the observation. Steps still saturate the motors,
so smooth trajectories are still the right next move.

### Finding: workspace boundary
A top-down grasp at (0.7, ±0.3, 0.4) is unreachable: 50 random IK restarts all stop at
8.66 mm. The wrist centre would be 0.81 m from the shoulder, past full extension.
The table workspace for top-down grasps is therefore limited to roughly x ≤ 0.65 m.

## 2026-11-14: Stage 2 (M5, M6, M7, M9)

Built: pick-and-place scene (MjSpec), Cartesian controller (IK at 100 Hz on top of joint
servos + gravity feedforward), minimum-jerk pose trajectories, scripted pick-and-place state
machine, seeded 100-episode evaluation with failure attribution. Results: docs/results_stage2.md.

### Finding: a held cube "slipped" ~1.5 mm/s, but it wasn't friction
**Observation:** during transport the cube sank ~4 mm relative to the fingers, while the TCP
height stayed constant (so not arm sag). The slip was identical at friction 1.0, 0.5 and 0.3.
**Cause:** MuJoCo contacts are soft constraints. With the default pyramidal friction cone and
impratio = 1, a sustained tangential load produces slow creep.
**Fix:** elliptic friction cone + impratio = 10 (MuJoCo's recommendation for grasping).
Median slip 3.0 mm → 0.4 mm over 100 episodes.

### Bug: the cube friction setting did nothing
**Symptom:** identical results for friction 1.0, 0.5 and 0.3 (suspicious in itself).
**Cause:** for two touching geoms with equal priority, MuJoCo uses the **maximum** of their
friction coefficients. The finger pads are 1.0, so the cube's value never mattered.
**Fix:** cube geom `priority=1`, so its friction governs its contacts.
**Lesson:** when a parameter sweep gives identical numbers, suspect the parameter isn't
connected before concluding the system is insensitive to it.

### Finding: the gripper is weak: ~1 N per finger
Measured with `mj_contactForce`: the gripper tendon pulls with 2.0 N, and the tendon couples
both fingers with coefficient 0.5, so each finger presses ~1.0 N. Friction capacity
2·μ·1 N vs 0.98 N weight predicts exactly the observed cliff: μ = 1.0 works, μ ≤ 0.5 fails 100%.

### Bug: stronger gripper made fast grasps fail (7% success)
**Symptom:** gripper stiffness ×10 at 1.0 m/s → 93/100 grasp failures, worse than ×1.
**Diagnosis:** finger gap reached 13.9 mm on a 40 mm cube; cube centre sank to 11 mm (below
its half-height). The fingers penetrated the cube.
**Cause:** I scaled kp ×10 but left kd unchanged. Damping ratio ζ = kd / (2√(kp·m)) fell by
√10 ≈ 3.2×, so the fingers closed underdamped and fast enough to tunnel into the contact.
**Fix:** scale kd by √(scale) to keep ζ constant. 7% → 100%.
**Lesson:** never change stiffness without asking what happens to the damping ratio.

## 2026-11-14: M8 collision-aware planning

Built: wall scene (random height 18-30 cm per episode), collision checker, RRT-Connect +
shortcutting, posture search using the 7th DoF, collision-checked Cartesian paths, planned
pick-and-place policy, comparison script. Results: docs/results_m8.md.

### Bug: inflated geom margins gave an inconsistent safety distance
**First design:** set every geom's `margin` to 2 cm and call `mj_collision`.
**Symptoms:** (a) the home pose was "in collision" with itself (links 5 and 7 are 2.3 cm
apart); (b) the hand 3.7 cm from the floor was flagged but a fingertip 7 mm away was not.
MuJoCo's effective margin depends on the shape types involved (pairs sum margins; some
colliders treat margins differently).
**Fix:** two separate checks. Robot-vs-obstacle uses `mj_geomDistance` (exact signed distance,
compared against the margin). Self-collision and the floor use `mj_collision` and count only
real overlap (dist < 0).

### Finding: the hand is 20 cm wide; four yaws are not equal near a wall
Hovering above a target 9 cm from the wall, the hand collided whenever its finger axis
pointed at the wall. The cube's 90° symmetry gives four equally valid grasp/place yaws;
the planner tries each and keeps the first collision-free one.

### Finding: the elbow, not the hand, hit the wall
For cubes 10-14 cm from the wall, every grasp yaw collided, but the collision was **link 5
(elbow) 13-42 mm inside the wall**, not the hand. IK's null-space term pulls toward the home
posture, which swings the elbow toward the wall when reaching down. Fix: search over arm
postures (IK from 40 random start postures, each used as its own null-space preference).
That is what the 7th joint is for.

### Bug: posture chosen at the top of a vertical move drifted into the wall on the way down
Choosing a clear posture above the cube and holding it during the descent still brought
links 5/6 to 18-19 mm of the wall (inside the 20 mm margin) near the bottom.
**Fix:** anchor the posture at the most constrained pose (the grasp / the place pose) and
trace the straight-line paths **upward** from there. Planned success on the first 12 seeds
went from 6/12 to 10/12.

### Test bug: the open gripper straddles a thin wall
A margin test assumed the fingertips would be ~1 cm above the wall; the checker said "free".
It was right: pointing down across the wall, the open fingers (8 cm gap) straddle the 3 cm
wall, 25 mm from each face. Rotating the hand 90° gives the intended geometry. Measured
fingertip offset: 8.9 mm below the TCP. The margin flips exactly at 20 mm clearance.

## 2026-11-14: Stage 3 perception (M10, M11, M12)

Built: fixed RGB-D camera (MjSpec), pinhole camera model with explicit OpenCV/OpenGL
conventions, colour + depth cube/target localization, sensor noise model, VisionPolicy that
feeds estimates into the unchanged Stage 2 / M8 policies. Results: docs/results_stage3.md.

### Verification of the camera model before using it
Back-projecting the renderer's depth for ground-truth floor pixels gives z = 0.00 mm, and for
cube pixels a maximum of 40.00 mm (the cube's top); the projected cube centre lands inside the
cube's ground-truth mask. That checks intrinsics, extrinsics, the y/z axis flip between
OpenGL and OpenCV, and the half-pixel centre convention in one go.

### Finding: the floor is a mirror
Menagerie's floor is reflective; the red cube's reflection is red too. Colour alone sees a
ghost cube below the table; depth puts those pixels at table height, so a 5 mm height cut
removes them. (Same problem as shiny tables for real cameras.)

### Bug: "highest points" top-face selection is biased under noise
**First method:** top face = points within 4 mm of the 95th-percentile height.
**Symptom:** with the "poor" noise model, height biased +6.8 mm and xy error 7.8 mm (median),
on perfect data it was exact.
**Cause:** selection bias. Thresholding noisy heights keeps the points whose noise pushed them
UP; depth noise acts along the viewing ray, and the camera looks down at ~50°, so "pushed up"
also means "pushed toward the camera": biased xy.
**Fix:** median-filter the depth image, then find the top face as the MODE of point heights
(mean-shift), with a band symmetric around it. Poor noise: xy 7.79 → 0.79 mm, height bias
+6.84 → +0.24 mm.

### Bug in the fix: smoothing the histogram smeared the top face
**First attempt at the mode:** histogram smoothed over ±4 mm, take the peak. On PERFECT depth
this gave 14.5 mm xy error and 44° yaw error.
**Cause:** the smoothing window was wider than the gap between the top-face spike and the
upper parts of the side faces, so the "peak" moved down into the sides and side points
entered the top-face set (wrecking the rectangle fit for yaw).
**Fix:** mean-shift from the fullest 1 mm bin: converges to the true mode without blurring it.

### Environment issue: headless rendering
Rendering needs `MUJOCO_GL=egl` on display-less Linux (servers, Colab). `ari/__init__.py`
now sets it automatically when there is no DISPLAY, and `tests/conftest.py` imports `ari`
first so it applies before `mujoco` is imported.

## 2026-11-14: Stage 4 imitation learning (M13, M14)

Built: observation/action spaces, demonstration recording, behaviour cloning in PyTorch,
a state-feedback expert, DAgger, parallel evaluation with a cube-moving disturbance.
Results: docs/results_stage4.md. Image-based policies (M15) are deferred to a GPU.

### Bug: replaying recorded actions did not reproduce the episode
Recorded the 100 Hz scripted expert, sampled at 20 Hz; replaying those actions failed to lift
the cube. Actions are deltas from the MEASURED pose, so the timing difference between a 100 Hz
command stream and a 20 Hz held command accumulates like an integrator.
**Fix:** the expert acts through the same 20 Hz executor the learner uses. Replay is now exact
(regression test).

### Bug: the expert dropped the cube through the 20 Hz interface (2 of 10 episodes)
Jumping to each new target 20 times a second jolts the stiff joint servos; the ~1 N grip from
Stage 2 let the cube slide out during the lift (gripper at 10.6 cm, cube at 7.5 cm).
**Fix:** interpolate the command linearly between decisions (`ActionExecutor`). 30/30 afterwards.
(This is the weak gripper from Stage 2 again; the stiffness decision is still open.)

### Finding: low validation loss, 18% success
First policy, cloned from 500 scripted demonstrations: validation MSE 0.027 (normalized), 4/20
rollouts. Failure traces: crawling at 0.2 mm/step mid-transport; pressing into the table after
release; flying off to z = 0.8 m.
**Cause:** the scripted expert is clock-driven (min-jerk timing, fixed pauses). For one observed
state the data holds many different actions; a memoryless network learns their mean.
**Fix:** a state-feedback expert (`ReactiveExpert`): action = f(state), no clock. 26/30 at once.

### Finding: hard thresholds in the expert are cliffs for the learner
Remaining failures all occurred while lowering the cube: the learner drifted 4-8 mm sideways and
froze. The expert's rule there: within 3 mm "lower", otherwise "climb back to carry height".
The expert never crossed that threshold itself, so the flaw was invisible in its own 100%.
**Fix:** continuous "funnel" rules (height target proportional to horizontal error).
Cloned policy 87% → 100%; the expert also got faster (8.7 → 6.8 s).

### Finding: DAgger's value only shows off-distribution
500 demos: nominal 100% without DAgger, 97% with (noise). With the cube moved mid-episode:
68% without, 99% with.

### Negative result: DAgger from a weak policy made it worse
100 demos: 71% → (DAgger) → 25%. Hypothesis 1: repeated stuck states swamp the data (54% of
samples were near-duplicates). Tested by removing repeats: 39%. Helps, does not explain.
Hypothesis 2: look-alike states with opposite labels (pre-grasp hover "down" vs post-place hover
"up"). The opposite labels are in the data; the predicted pattern in failures is weak (47% vs 32%
across separation groups, small groups). Left open with a proposed test (results_stage4.md).
**Lesson:** I had a plausible story after the first measurement and it was wrong. The second story
is also plausible and is not confirmed. Both are recorded as such.

### Environment: import order again
`scripts/bc_demo.py` imported `mujoco` before `ari` and lost the headless-renderer setting from
Stage 3. Scripts that render must import `ari` first.

## 2026-11-14: M15 learning from images (CPU-scale), and the DAgger question resolved

Built: wrist camera, teacher-student image policy (two small CNNs with spatial softmax),
image demonstration/DAgger collection, LeRobot ACT script (smoke-tested), Colab notebook.
Results: docs/results_m15.md.

### Result: the image student grasps but cannot place (43% nominal, 4% with the cube moved)
It lifts the cube in 95/100 episodes; 45 failures are misplacements. While carrying, the wrist
camera sees the cube, not the target; the front camera shows the target ~6 px wide. Not fixed here.

### Environment: software rendering is 12× slower with shadows and reflections
200 ms per 64×64 image by default, 16 ms with both off. Off for the student's images.
Also an 8× mis-estimate: I budgeted ~4 min per training epoch from a timing taken
while other jobs were running; uncontended it was ~30 s. Measure on an idle machine.

### The DAgger question from Stage 4: three wrong or partial stories, then a dissection
Symptom in both the state-based (100 demos) and image policies: DAgger rounds made the policy worse.
1. *Repeated stuck states swamp the data.* Removing repeats: 25% → 39%. Minor.
2. *Look-alike states with opposite labels.* Consistent with the data, never confirmed.
3. *States on the expert's decision cliffs.* Measured: 4.0% of a weak learner's DAgger states vs
   0.4% of demos. Removing them: still 35%. A real correlation that was not the cause.
4. **Dissection instead of a fourth guess:** retrained once, found three frozen states, printed
   the nearest training samples. The policy's motion output was right (≈0) and the state sat a few
   mm outside the expert's grasp trigger: a precision failure. Training error by configuration:
   0.0007 (500 demos) / 0.0053 (100) / 0.0079 (100 + DAgger, 40 epochs).
   Intervention: same data, 200 epochs → error 0.0011, success 48% → 70% (60 episodes).
   Full rerun with 160 epochs per round: nominal 70%, disturbed 67% (plain cloning: 71% / 47%).
**Conclusion:** I held the training budget fixed while DAgger made the dataset larger and more
varied. The fit lost precision, and this task fails below ~3 mm precision at the grasp.
**Lessons:** (a) three plausible explanations in a row were wrong or minor; only the direct look
at failing states found it. (b) A correlation that matches the hypothesis (cliffs) is not a cause
until an intervention says so. (c) When adding data, check that training error has not gone up.

### Negative result that stands: DAgger on the image student
3 fine-tuning epochs: 23/60 → 1/60. 8 epochs: 24/60 → 2/60. Training error after aggregation
stayed 5-10× above the pre-DAgger level. Probably the same under-fitting, not shown. Open.

### Bug: training log crashed on a Path argument
`json.dumps` of the argument dict failed after the model was saved. Fixed with `default=str`.

## 2026-11-14: Stage 5 reinforcement learning (M16, M17, M18)

Built: Gymnasium environments (reach, pick-and-place) with potential-based shaping, PPO from
scratch with persistent worker processes, residual policy on the frozen clone, RL evaluation
through the Stage 4 harness, learning-curve plot. Results: docs/results_stage5.md.

### Performance: the simulator was too slow for RL, and 95% of the time was my own IK
123 policy steps/s; profiling showed `solve_ik` (pure-Python DH chain, 5 solves per decision).
Closed-form DH link transforms and one pass for pose + Jacobian: 262 steps/s. IK once per
decision with joint-space interpolation: 634 steps/s per core. Kinematics tests (FK and Jacobian
against MuJoCo) guard the rewrite. Baselines re-measured under the new executor (clone 71% → 66%,
within noise).

### PPO verified before use
Pendulum-v1: about −1100 → about −200 in 150k steps. Only then pointed at the robot.

### Finding: exploration noise alone raised the clone from 66% to 79%
First residual iterations (critic warm-up, policy frozen) already showed ~82% training success.
The clone's failures are stalls a few mm from the grasp trigger; 1.5 mm of jitter breaks them.
A "noise, no learning" baseline was added to the evaluation; it accounts for half of RL's gain.

### Finding: reward specification shows up in the result
Residual RL made placement less accurate (4 mm → 9-12 mm) while raising success and speed. The
reward is flat inside the 3 cm target zone and charges per step.

### Reward bug: idle income under potential-based shaping
With γ = 0.99, a step that changes nothing earns γΦ − Φ = −0.01·Φ. My Φ was negative (around −1),
so standing still paid about +0.01 per step, cancelling the −0.01 time penalty. The theory
(policy invariance) is intact; the learning signal is not. Fix: offset Φ to be non-negative.
Did NOT rescue from-scratch pick-and-place (still 0%), so it was a real flaw but not the cause.
The residual results were trained with the original reward; the offset is an option, default off,
so they stay reproducible.

### Negative result: pick-and-place from scratch, 0% in 600k and 500k steps
The policy does not even approach the cube (closest approach ~30-34 cm), although the reach task
is learned in 150k steps with the same PPO. Cause not established; candidates listed in
results_stage5.md. "Push" was dropped from the curriculum.

### Test detail: the action space clips
The residual-equals-clone test failed on a 0.0001 rad difference: the clone asked for −0.1501 rad
and the Gymnasium action space clips at −0.15. The test now compares against the clipped clone.

## 2026-11-15: Stage 6 language and task-level autonomy (M19, M20; M21 not attempted)

Built: multi-object scene (3 cubes, 3 targets) that every earlier skill runs in unchanged via
`focus(cube, target)`; instruction grammar with held-out splits; bag-of-words and GRU parsers;
a language-conditioned policy trained by behaviour cloning only; a task planner with clause
splitting, plan validation, post-condition monitoring, bounded retries, a fallback skill and
fault injection. Results: docs/results_stage6.md.

### Test-design flaw: a held-out split that could not be solved
First held-out pairs: blue→red and yellow→green. Both parsers: 100% seen, 48.2% new pairing.
With blue→red absent from training, {red, blue} always meant red→blue, so word order carried no
information in the training set and neither model learned it. Changed the held-out pairs to
yellow→green and blue→blue and added an "order only" split (red→blue vs blue→red). After that:
bag of words 10-52% on order only, GRU 100%. Lesson: before blaming a model for failing a
generalization test, check that the training data determines the answer.

### Wrong conclusion from one seed: "word order cannot be learned from actions"
Seed 0 of the language policy reads "on the blue target put the red cube" as blue→red every
time (grounding 93.5%, later 96.4%), while the same GRU trained as a parser with labels reads it
correctly. I wrote this up as a finding and reported the end-to-end policy at 108/150 against
149/150 for parser + skill. Seeds 1 and 2, run to put a number on the variance, scored 144 and
146 and are 100% correct on the order-only split. The finding was a property of one run. All seven runs are in
results_stage6.md; the supported statement is "the outcome varies strongly between seeds".

### Negative result, partly explained: pretrained text encoder gives 0% grounding (seed 0)
Initializing the policy's text encoder from the parser: cube attention correct, target
attention a consistent wrong assignment, 2/150.
- Hypothesis 1, saturated attention at initialization. Measured: sentence-vector norm 9.2 vs
  2.6; initial max attention weight 0.88 vs 0.67. Added LayerNorm on the sentence vector, query
  weights ×0.1, scale 1: initial max weight 0.40. From-scratch seed 0 improved (93.5% → 96.4%);
  the pretrained run did not (0%). Rejected as the cause; the change is kept.
- Hypothesis 2, an unlucky seed. Pretrained seed 1: 100% grounding, 148/150. Confirmed that it
  is seed-dependent.
- Observation: in every run the grounding after epoch 1 is within a few points of the final
  value (95.9 → 96.4, 100 → 100, 97.8 → 100, 0 → 0, 100 → 100).
- Hypothesis 3, the attention commits before the motion network has learned what the target is
  for. Test: learning rate ×0.1 for text encoder and attention, same seed 0: 100% grounding,
  148/150. The same change on from-scratch seed 0: still 96.4%, 124/150. So it fixed the
  pretrained failure in the one case tried and is not a general cure.

### Finding: imperfect grounding and poor motion come together
Runs ending at 96% grounding: 23 and 19 motion failures with the right objects. Runs at 100%:
0, 0, 0, 2. Motion error given the true objects differs by only about 1.4×. Plausibly the 4% of
samples that feed the motion network the wrong object; not tested directly.

### Training-speed trap: two PyTorch jobs on two cores
Two training runs in parallel, each with 2 threads on a 2-core machine, ran more than 10× slower
than one after the other (2.4 min per epoch against 8 s). Run them in sequence.

### Finding: the end-to-end policy can match parser + skill, but not reliably
M19, 150 episodes: parser + Stage 4 skill 149; end-to-end 108, 144, 146 (three seeds); with a
pretrained text encoder 2 or 148. Because the attention is inspectable, each failure can be
attributed to grounding or to motion.

### Bug: fault injector crashed on bumps
`Faults._free_spot` unpacked `region` even when sampling near a point (`region=None`). Unpack
only when `near is None`.

### Finding: the learned skill fails just outside its training envelope, twice
1. bc_dagger failed 3 of 45 first runs in the multi-object scene; all three had a cube 9-10 cm
   from a target, below the 12 cm minimum of the Stage 4 scenes. The multi-object sampler now
   enforces `min_separation`.
2. With faults and monitoring it recovers 43% of faulted tasks (scripted and reactive skills:
   100%). After a fault the cube lies somewhere the skill never trained on, and retrying the
   same skill fails the same way. Added `fallback`: learned skill first, classical skill for
   retries: 100% of faulted tasks recovered.

### Refactor: executors can start mid-episode
`rollout` split into `run_policy` (run from the current state) and `rollout` (reset + run +
failure attribution). `ActionExecutor` now initializes its controller from the current joint
positions and gripper command instead of assuming the home pose. Stage 4/5 tests unchanged.

### M21 (VLA fine-tuning): not attempted
Needs a GPU and pretrained weights; neither is available in the build environment.

## 2026-11-15: Stage 7 benchmark, report, demo video (M22, M23)

Built: `benchmark.py` (13 conditions, sensing errors, latency, out-of-area starts, severity
sweeps), `scripts/benchmark.py` (resumable runner), `scripts/benchmark_report.py` (heatmap,
sweeps, tables), `scripts/make_demo_video.py` (the demo rendered from live episodes),
`docs/report.md`. Results: docs/results_stage7.md.

### Design: where to inject sensing errors
Every policy reads the cube through `env.cube_pose()` (directly, or via the observation
vector), while success, failure attribution and the cube-shift disturbance read the simulator
state directly. Overriding that one method perturbs what all six policies perceive without
touching how they are judged. A test checks both halves.

### Design: condition strengths were fixed before the full run, by physics where possible
A 20-episode probe with two policies was used to check that each condition does what it says.
It showed that a 0.2 kg cube is exactly at the gripper's limit (2 x friction x 1 N = 2 N against
1.96 N: 0/20), the same phenomenon as friction 0.5. Keeping both would have spent two columns on
one fact, so the mass condition was set to 0.15 kg and the 0.2 kg result is reported as a
footnote (0/100 for two policies).

### Finding: the reactive policy had the cube size written into it
5 cm cube: 0/100. `ReactiveExpert.half = 0.02` is a constant, and "holding" is decided from the
finger opening against it. Told the size: 100/100. I wrote that policy in Stage 4 and tested it
only on 4 cm cubes; the benchmark found the assumption in its first run.

### Finding: the clone beats its teacher on that same cube; DAgger does not
500-demo clone 78%, DAgger clone 8% (it lifts and never releases), teacher 0%.

### Finding, then design flaw: latency and the relative action interface
100 ms of latency cost every learned policy 26-44 points and the hand-written ones nothing. I
first logged this as "cause not established". The sweep then showed the hand-written policies
collapsing too, at 200 ms (scripted 11%, reactive 44%), which an open-loop trajectory should
not do under a uniform delay. That pointed at the interface: an action is a step relative to
the pose measured when it is APPLIED, so a late step is added to a pose the policy never saw
and the target overshoots by the distance travelled during the delay. Test: same delay, late
actions keep their original pose target. Result: all six policies back at their undelayed level
at both 100 and 200 ms. The Stage 4 decision ("relative to the measured pose, so the servo lag
cancels") assumed measurement and action happen at the same instant. Not changed in the
default interface (every trained model depends on it); recorded as the first thing to fix.

### Finding: the image policy does not extrapolate
5 cm outside its demonstration area: 7% (nominal 42%); the state-based 500-demo clone loses 8
points there.

### Process mistakes while running the benchmark
Two background jobs writing the same results file would have overwritten each other's cells;
the second was restarted with its own output file and merged (the four cells run twice gave
identical counts). A "wait until the other job ends" loop matched its own command line and
never ended; 12 minutes lost.

### Video: two rendering problems
The software renderer draws striped artefacts at the edge of the shadow map at 1280x720;
changing shadow clip, scale and scene extent did not remove them, so shadows are off in the
video. And the task planner's `go_home` calls the frame callback at the 100 Hz control rate
while skills call it at 20 Hz, which would have played the return-to-home motion five times too
slowly; frames are now taken by simulated time (one per 50 ms), whoever calls.

### Not done
ROS 2 wrapper; M21 (VLA); full-scale image policy. Listed in docs/report.md section 7.
