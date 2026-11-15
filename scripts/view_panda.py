"""M0/M2: open the interactive viewer and sweep each arm joint with a sine wave.

Run on macOS:  mjpython scripts/view_panda.py
Run on Linux:  python scripts/view_panda.py

Every 100 physics steps it prints the hand position. In M3 you reproduce those
numbers with your own forward kinematics.
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from ari.control.joint import arm_actuator_ids
from ari.robot.model import load_panda, reset_to_home

AMPLITUDE = 0.4   # rad, try changing these
FREQUENCY = 0.25  # Hz

model, data = load_panda()
reset_to_home(model, data)
q_home = data.ctrl[:7].copy()
act = arm_actuator_ids(model)
hand = model.body("hand").id

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        start = time.time()
        # Each joint gets its own phase so you can see them move independently.
        phases = np.arange(7) * np.pi / 4
        data.ctrl[act] = q_home + AMPLITUDE * np.sin(2 * np.pi * FREQUENCY * data.time + phases)
        mujoco.mj_step(model, data)
        if step % 100 == 0:
            print(f"t={data.time:6.2f}s  hand xyz = {np.round(data.xpos[hand], 4)}")
        step += 1
        viewer.sync()
        # Keep roughly real time.
        time.sleep(max(0.0, model.opt.timestep - (time.time() - start)))
