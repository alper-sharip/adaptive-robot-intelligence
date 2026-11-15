"""M1: take the Panda model apart and print what it is made of.

Run:  python scripts/inspect_model.py
"""

import mujoco
import numpy as np

from ari.robot.model import load_panda, reset_to_home

JOINT_TYPES = {0: "free", 1: "ball", 2: "slide", 3: "hinge"}
TRN_TYPES = {0: "joint", 3: "tendon"}

model, data = load_panda()
reset_to_home(model, data)
np.set_printoptions(precision=3, suppress=True)

print("=== Sizes ===")
print(f"nq   (position coordinates, len(data.qpos)) : {model.nq}")
print(f"nv   (velocity coordinates, len(data.qvel)) : {model.nv}")
print(f"nu   (actuators, len(data.ctrl))            : {model.nu}")
print(f"njnt (joints)                               : {model.njnt}")
print(f"nbody(bodies, including 'world')            : {model.nbody}")
print(f"timestep                                    : {model.opt.timestep} s "
      f"({1 / model.opt.timestep:.0f} Hz physics)")

print("\n=== Kinematic tree (body <- parent) ===")
for b in range(model.nbody):
    parent = model.body(model.body_parentid[b]).name
    print(f"  {model.body(b).name:13s} <- {parent:13s}  offset in parent: {model.body_pos[b]}")

print("\n=== Joints ===")
for j in range(model.njnt):
    jt = model.joint(j)
    print(f"  {jt.name:14s} {JOINT_TYPES[model.jnt_type[j]]:6s} body={model.body(model.jnt_bodyid[j]).name:12s} "
          f"axis={model.jnt_axis[j]} range={model.jnt_range[j]}")

print("\n=== Actuators ===")
for a in range(model.nu):
    act = model.actuator(a)
    trn = TRN_TYPES.get(model.actuator_trntype[a], "?")
    target = (model.joint(model.actuator_trnid[a, 0]).name if trn == "joint"
              else model.tendon(model.actuator_trnid[a, 0]).name)
    kp = model.actuator_gainprm[a, 0]
    kd = -model.actuator_biasprm[a, 2]
    print(f"  {act.name:10s} drives {trn} '{target}'  kp={kp:8.3f} kd={kd:6.1f} "
          f"ctrlrange={model.actuator_ctrlrange[a]} force limit=±{model.actuator_forcerange[a, 1]:.0f}")

print("\n=== State at 'home' keyframe ===")
print("qpos:", data.qpos)
hand = model.body("hand").id
print("hand position (world, m):", data.xpos[hand])
print("hand orientation (world rotation matrix):\n", data.xmat[hand].reshape(3, 3))

print("\nQuestions to think about:")
print(" 1. Why are there 9 joints but only 8 actuators?")
print(" 2. The arm actuators have ctrlrange equal to the joint range. What does ctrl mean, then?")
print(" 3. Which column of the hand's rotation matrix is the direction the gripper points?")
