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
