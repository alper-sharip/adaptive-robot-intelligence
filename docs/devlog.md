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
