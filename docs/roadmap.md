# Roadmap

Status legend: ⬜ Planned · 🟨 In progress · ✅ Implemented · 🧪 Tested

## Stage 1: Simulation & kinematics
- ✅ M0 Environment setup
- ✅ M1 Inspect the Panda model
- 🧪 M2 Joint control (PD servos, gravity compensation, step-response metrics)
- 🧪 M3 Forward kinematics from scratch (DH), verified against MuJoCo
- 🧪 M4 Inverse kinematics (damped least squares + null space), verified on random targets

## Stage 2: Manipulation
- 🧪 M5 Scene with cube + target (MjSpec); Cartesian controller (IK + gravity feedforward)
- 🧪 M6 Minimum-jerk pose trajectories
- 🧪 M7 Scripted pick-and-place: 100/100 at defaults, stress-tested (docs/results_stage2.md)
- 🧪 M8 Collision-aware planning: RRT-Connect + posture search; 96% clean success vs 60% heuristic (docs/results_m8.md)
- 🧪 M9 Seeded randomized environment + evaluation with CIs and failure attribution

## Stage 3: Perception
- ⬜ M10 Simulated RGB-D camera, intrinsics/extrinsics
- ⬜ M11 Object segmentation + 3D localization (error measured vs ground truth)
- ⬜ M12 Vision-driven pick-and-place (classical baseline v1)

## Stage 4: Imitation learning
- ⬜ M13 Demo collection in LeRobotDataset format
- ⬜ M14 Behaviour cloning (state-based), own PyTorch implementation
- ⬜ M15 Image-based policy (ACT or Diffusion Policy via LeRobot), compared vs baseline

## Stage 5: Reinforcement learning
- ⬜ M16 Gymnasium env: reach
- ⬜ M17 PPO/SAC: reach → push → pick
- ⬜ M18 Residual RL on top of BC

## Stage 6: Autonomous intelligence
- ⬜ M19 Language-conditioned policy
- ⬜ M20 Planner over verified skills + failure recovery
- ⬜ M21 VLA fine-tuning investigation

## Stage 7: Evaluation & portfolio
- ⬜ M22 Generalization benchmark
- ⬜ M23 Report, demo video, README, resume bullets
