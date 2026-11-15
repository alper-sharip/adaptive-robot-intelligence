"""M13: what a learned policy observes and what it outputs.

These two definitions are the most important design decisions in Stage 4. Everything the
network can ever know is in the observation; everything it can ever do is in the action.

Observation (state-based, 19 numbers). Object poses come from the simulator here; Stage 3
showed the camera can supply them at the start of an episode, and M15 replaces this vector
with images.
    tcp position (3)             where the gripper is
    tcp yaw as sin, cos (2)      angles as sin/cos avoid a jump at +/-180 deg
    finger opening (1)           tells the policy whether the grasp has closed
    cube position (3)
    cube yaw as sin 4a, cos 4a (2)   a cube repeats every 90 deg; 4a makes equal poses equal inputs
    target xy (2)
    cube - tcp (3), target - tcp (2), cube - target xy... (relative vectors: what the
                                 policy actually needs; giving them directly helps a small MLP)

Action (5 numbers), issued at 20 Hz:
    dx, dy, dz    where to send the gripper, RELATIVE to where it is measured to be now
    dyaw          rotation about the vertical, relative to the measured yaw
    gripper       1 = closed, 0 = open
The gripper always points straight down, so one angle describes its orientation.

Why relative to the MEASURED pose? The command becomes `measured + delta`, so the policy is
closed-loop: if the arm is not where it should be, the same delta still makes sense. The
joint servos lag their command by ~0.1 s (Stage 1), so the expert's delta includes that lag;
because training and execution compute it the same way, it cancels.
"""

from __future__ import annotations

import numpy as np

from ari.robot.transforms import make_T, rot_x, rot_z

GRIPPER_DOWN = rot_x(np.pi)
POLICY_DT = 0.05          # s: the policy acts at 20 Hz; the controller runs at 100 Hz underneath
OBS_DIM = 20
ACT_DIM = 5


def tcp_yaw(T: np.ndarray) -> float:
    """Yaw such that T's rotation = rot_z(yaw) @ GRIPPER_DOWN (valid for top-down poses)."""
    R = T[:3, :3] @ GRIPPER_DOWN.T
    return float(np.arctan2(R[1, 0], R[0, 0]))


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def make_obs(env) -> np.ndarray:
    T = env.tcp_pose()
    p, yaw = T[:3, 3], tcp_yaw(T)
    cube, cyaw, tgt = env.cube_pose()[:3, 3], env.cube_yaw(), env.target_pos()[:2]
    return np.concatenate([
        p, [np.sin(yaw), np.cos(yaw)], [env.finger_width()],
        cube, [np.sin(4 * cyaw), np.cos(4 * cyaw)], tgt,
        cube - p, tgt - p[:2], cube[:2] - tgt,
    ]).astype(np.float32)


def action_from_target(T_now: np.ndarray, T_target: np.ndarray, gripper_closed: bool) -> np.ndarray:
    """Encode the expert's commanded pose as a delta from the measured pose."""
    dp = T_target[:3, 3] - T_now[:3, 3]
    dyaw = wrap(tcp_yaw(T_target) - tcp_yaw(T_now))
    return np.array([*dp, dyaw, float(gripper_closed)], dtype=np.float32)


def target_from_action(T_now: np.ndarray, action: np.ndarray):
    """Decode: (4x4 TCP target, gripper_closed)."""
    yaw = tcp_yaw(T_now) + float(action[3])
    T = make_T(rot_z(yaw) @ GRIPPER_DOWN, T_now[:3, 3] + action[:3])
    return T, bool(action[4] > 0.5)
