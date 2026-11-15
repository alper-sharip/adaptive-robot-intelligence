# Adaptive Robot Intelligence

An autonomous robotic manipulation system built in simulation, from first principles up:
kinematics and control → motion planning → perception → imitation learning →
reinforcement learning → language-conditioned behaviour.

**Minimum viable system (target):** a simulated Franka Panda picks a cube from a random
position using only camera input and places it in a target zone. Scripted, imitation-learned
and RL policies are compared on a fixed, seeded evaluation set.

## Status

| Milestone | Status | Evidence |
|---|---|---|
| M0 Environment setup | ✅ Implemented | `scripts/view_panda.py` |
| M1 Model inspection | ✅ Implemented | `scripts/inspect_model.py` |
| M2 Joint control | 🧪 Tested | `tests/test_control.py`, step-response plots |
| M3 Forward kinematics (from scratch, DH) | 🧪 Tested | matches MuJoCo to 1e-9 m over 200 random configs |
| M4 Inverse kinematics (damped least squares) | 🧪 Tested | 100% success on 100 random reachable targets, mean error 0.05 mm |
| M5+ Scene, trajectories, pick-and-place, … | ⬜ Planned | see `docs/roadmap.md` |

✅ Implemented = code exists and runs · 🧪 Tested = verified by automated tests or measured results

## Results so far

**Reaching Cartesian targets** (gripper pointing down, IK solution executed in physics for 1.5 s):

| Target (m) | IK iterations | IK error | Error in sim (no gravity comp) | Error in sim (gravity comp) |
|---|---|---|---|---|
| (0.5, 0.0, 0.4) | 4 | 0.06 mm | 6.1 mm | 0.06 mm |
| (0.5, 0.3, 0.2) | 5 | 0.09 mm | 7.7 mm | 0.09 mm |
| (0.4, −0.3, 0.15) | 6 | 0.09 mm | 6.5 mm | 0.09 mm |
| (0.3, 0.0, 0.6) | 9 | 0.10 mm | 2.9 mm | 0.10 mm |
| (0.6, 0.0, 0.1) | 7 | 0.03 mm | 8.8 mm | 0.03 mm |

The IK math is accurate; without gravity compensation, the PD servos sag under the arm's
weight by several millimetres. Adding a gravity feedforward torque removes that error.

![IK demo](outputs/ik_demo.png)

## Quickstart (macOS / Linux)

```bash
git clone https://github.com/google-deepmind/mujoco_menagerie.git   # next to this repo
cd adaptive-robot-intelligence
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

python scripts/inspect_model.py                   # what the robot is made of
mjpython scripts/view_panda.py                    # interactive viewer (use `python` on Linux)
python scripts/joint_step_response.py             # control experiment -> outputs/
python scripts/ik_demo.py --gravity-comp          # IK + execution -> outputs/
pytest -s                                         # 19 tests
```

If Menagerie is somewhere else: `export ARI_MENAGERIE=/path/to/mujoco_menagerie`.

## Repository layout

```
src/ari/
  robot/      model loading, transforms, forward kinematics, Jacobian, IK
  control/    joint-space PD control, gravity compensation, step-response metrics
scripts/      runnable experiments
tests/        automated checks (kinematics vs MuJoCo, IK success rate, control behaviour)
docs/         roadmap, development log, concept notes
outputs/      generated plots and images
```

## Design decisions

- **MuJoCo + Franka Panda (MuJoCo Menagerie).** Accurate contact, fast on CPU, the most-studied arm in manipulation research.
- **Kinematics written from Franka's DH parameters, not read from MuJoCo.** Agreement between two independent robot descriptions is a real check, not a tautology.
- **ROS 2 deferred.** No engineering need inside a single-process simulation; it may be added later as an integration layer.
