"""Loading the Franka Panda model and naming its parts.

Everything else in the project gets the robot through `load_panda()`, so the
location of MuJoCo Menagerie is decided in exactly one place.
"""

from __future__ import annotations

import os
from pathlib import Path

import mujoco
import numpy as np

# Where MuJoCo Menagerie lives. Override with:  export ARI_MENAGERIE=/path/to/mujoco_menagerie
_REPO_ROOT = Path(__file__).resolve().parents[3]
MENAGERIE_DIR = Path(os.environ.get("ARI_MENAGERIE", _REPO_ROOT.parent / "mujoco_menagerie"))
PANDA_SCENE = MENAGERIE_DIR / "franka_emika_panda" / "scene.xml"

# Names as they appear in the Menagerie MJCF file.
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]
ARM_ACTUATORS = [f"actuator{i}" for i in range(1, 8)]
GRIPPER_ACTUATOR = "actuator8"  # one actuator drives both fingers through a tendon
HAND_BODY = "hand"

# Distance from the hand frame origin to the point midway between the fingertips
# (the "tool centre point", TCP), along the hand's z-axis. Franka's published value.
TCP_OFFSET = 0.1034


def load_panda(scene: Path | str = PANDA_SCENE) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load the Panda scene. Returns (model, data).

    model: the fixed description (links, joints, masses, actuator gains).
    data:  the changing state (joint angles, velocities, contacts, controls).
    """
    scene = Path(scene)
    if not scene.exists():
        raise FileNotFoundError(
            f"Panda scene not found at {scene}.\n"
            "Clone MuJoCo Menagerie next to this repo, or set ARI_MENAGERIE."
        )
    model = mujoco.MjModel.from_xml_path(str(scene))
    data = mujoco.MjData(model)
    return model, data


def reset_to_home(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Put the robot in the 'home' keyframe defined in the MJCF (arm bent, gripper open)."""
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    mujoco.mj_forward(model, data)


def arm_qpos_indices(model: mujoco.MjModel) -> np.ndarray:
    """Indices into data.qpos for the 7 arm joints (in order joint1..joint7)."""
    return np.array([model.joint(name).qposadr[0] for name in ARM_JOINTS])


def arm_dof_indices(model: mujoco.MjModel) -> np.ndarray:
    """Indices into data.qvel / Jacobian columns for the 7 arm joints."""
    return np.array([model.joint(name).dofadr[0] for name in ARM_JOINTS])


def arm_joint_limits(model: mujoco.MjModel) -> np.ndarray:
    """(7, 2) array of [lower, upper] joint limits in radians."""
    return np.array([model.joint(name).range for name in ARM_JOINTS])
