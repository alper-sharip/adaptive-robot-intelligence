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
- 🧪 M10 Simulated RGB-D camera, intrinsics/extrinsics; verified against renderer depth to < 0.1 mm
- 🧪 M11 Colour + depth localization: cube 0.3-0.6 mm, target ~1 mm (median), with a sensor noise model
- 🧪 M12 Vision-driven pick-and-place: matches ground-truth success in all 6 conditions (docs/results_stage3.md)

## Stage 4: Imitation learning
- 🧪 M13 Demonstration recording (clock-driven and state-feedback experts); .npz with LeRobot column names (conversion to LeRobotDataset happens in M15)
- 🧪 M14 Behaviour cloning + DAgger in plain PyTorch: 100% nominal, 99% with the cube moved mid-episode (docs/results_stage4.md)
- 🟨 M15 Image-based policy: own CNN student 🧪 tested at CPU scale (43% nominal, 4% disturbed; docs/results_m15.md). Full-scale GPU run and LeRobot ACT: scripts and Colab notebook ready, ACT smoke-tested only, not trained

## Stage 5: Reinforcement learning
- 🧪 M16 Gymnasium environments (reach, pick-and-place), potential-based shaping, `check_env` passes
- 🧪 M17 Own PPO (verified on Pendulum-v1): reach from scratch 100% (3 seeds); pick-and-place from scratch 0% (negative result); push not attempted
- 🧪 M18 Residual RL on the 100-demo clone: 66% → 92-93% nominal (2 seeds), 43% → 64-73% disturbed (docs/results_stage5.md)

## Stage 6: Autonomous intelligence
- 🧪 M19 Language-conditioned policy: multi-object scene, generated instruction grammar with held-out splits; GRU parser + Stage 4 skill 149/150; end-to-end language policy (behaviour cloning only) 108, 144, 146 of 150 over three seeds, every failure attributed to grounding or motion (docs/results_stage6.md)
- 🧪 M20 Planner over verified skills + failure recovery: plan validation before motion (5 refusals, no misunderstood instruction executed), post-condition monitoring with retries, learned skill with classical fallback: 95% of two-step tasks with injected faults vs 35-45% without monitoring
- ⬜ M21 VLA fine-tuning investigation: not attempted (needs a GPU and pretrained weights)

## Stage 7: Evaluation & portfolio
- 🧪 M22 Generalization benchmark: 6 policies × 13 conditions × 100 seeded episodes, severity sweeps for sensing jitter, sensing offset and latency; the two flaws it found (latency through the action interface, the reactive policy's fixed cube size) fixed and re-measured, placement precision, failure types (docs/results_stage7.md)
- ✅ M23 Final report (docs/report.md), demo video rendered from live simulation (outputs/demo.mp4), README front page
- ⬜ ROS 2 wrapper (optional): not done

## After the stages
- 🧪 Latency fix: actions anchored to the observed pose; 100 ms latency now costs no policy anything (docs/results_stage7.md)
- 🧪 Camera loop: every single-object policy on camera input, cube tracked every 50 ms (wrist camera first, front camera backup); strong policies unchanged, the two weakest lose 18-19 of 300 paired episodes (docs/results_camera.md)
- ⬜ Perception for the multi-object / language layer
- ⬜ Latency compensation
- ✅ Interactive slider control with live kinematics: DH table, link transforms, FK checked against MuJoCo, IK convergence, Jacobian, closed-form elbow angle, elbow swivel in the null space (scripts/slider_control.py)
