# Concept notes: M1–M4

Read this alongside the code. Each section links a concept to where it lives.

## 1. Model vs data (M1), `robot/model.py`
- `MjModel` is everything that never changes: link lengths, masses, joint axes, actuator gains.
- `MjData` is the state: `qpos` (joint positions), `qvel`, `ctrl` (commands), plus everything
  derived from them: `xpos`/`xmat` (body poses), contacts, forces.
- `mj_forward` recomputes derived quantities from the state without advancing time.
  `mj_step` advances time by one timestep (2 ms here).
- `nq = 9`: 7 arm joints + 2 finger slides. `nu = 8`: the two fingers share one actuator
  through a tendon, so they always move together.

## 2. Coordinate frames and transforms (M3), `robot/transforms.py`
- A pose is a 4×4 matrix `T = [[R, p], [0, 1]]`. R's columns are the frame's x, y, z axes
  expressed in world coordinates; p is its origin.
- Chaining: `T_world_hand = T_world_link1 @ T_link1_link2 @ ... @ T_link7_hand`.
- At "home", the hand's rotation matrix has third column (0, 0, −1): the gripper's z-axis
  points straight down.

## 3. Forward kinematics (M3), `robot/kinematics.py`
- DH parameters describe each link with 4 numbers (a, d, alpha, theta). Theta is the joint angle.
- FK = multiply the 7 link transforms, then the fixed flange and hand offsets.
- **Try:** pick q, run `hand_pose(q)`, then check it in the viewer by setting the same joints.

## 4. The Jacobian (M4), `robot/kinematics.py: tcp_jacobian`
- It maps joint velocities to TCP velocity: `[v; ω] = J(q) q̇`. It is 6×7.
- Column i for a revolute joint: linear part `zᵢ × (p − oᵢ)`, angular part `zᵢ`.
- When J loses rank (a **singularity**, e.g. arm fully stretched), some TCP directions
  become impossible to move in, and inverting J produces huge joint velocities.
- Tested three ways: against MuJoCo's `mj_jac`, against finite differences, and implicitly by IK working.

## 5. Inverse kinematics (M4), `robot/ik.py`
- No closed form for a general 7-DoF arm, so we iterate: measure error, take a step `dq`, repeat.
- **Damped least squares:** `dq = Jᵀ(JJᵀ + λ²I)⁻¹ e`. λ = 0 is the pseudo-inverse (fast,
  unstable near singularities); larger λ is slower but safe.
- **Redundancy:** 7 joints, 6 constraints, so a 1-D family of solutions (the "elbow" can
  swing while the hand stays still). The null-space term uses that freedom to stay near a
  comfortable posture.
- **Experiments to run:**
  - `damping` 0.001 / 0.05 / 0.5: count iterations, and try a near-singular target.
  - `nullspace_gain` 0 vs 0.05 vs 0.5: compare the resulting joint angles for the same target.
  - Start from different `q_init`: do you get different solutions? Why?

## 6. PD control and gravity (M2), `control/joint.py`
- Servo torque: `τ = kp(q_target − q) − kd·q̇`, clipped to the motor limit.
- At rest under gravity: `kp·error = τ_gravity`, so error ≈ τ_gravity / kp. Joints carrying
  the most arm weight (2 and 4) sag most.
- Damping ratio ζ = kd / (2√(kp·I)). Too low gives oscillation; too high gives slow response.
- These servos are heavily overdamped, so they behave like a first-order system with time
  constant kd/kp (0.1 s on every joint). See the corrected entry in devlog.md.
- Gravity compensation adds `τ_ff = g(q)`, so the PD only corrects the error.
- **Experiments:** `--kp-scale 0.1`, `--kd-scale 0.2`, `--gravity-comp`. Predict before you run.

## Check your understanding (answer these before M5)
1. Why does joint 1 never sag under gravity, whatever kp is?
2. Why is the step-response rise time almost the same for all joints despite different kp?
3. If IK returns `success=False` with `pos_error = 8.7 mm`, how do you tell "unreachable" from "solver stuck"?
4. What would go wrong if the TCP offset were set to 0?
