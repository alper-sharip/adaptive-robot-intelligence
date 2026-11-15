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
