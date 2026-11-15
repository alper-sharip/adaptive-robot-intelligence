"""Smooth point-to-point trajectories (M6).

Why not just command the goal directly? Stage 1 showed that a step command makes
the servos demand ~1350 Nm and saturate at the torque limit. A smooth path keeps
demanded torques small and the motion predictable.

Minimum-jerk profile: among all paths from rest to rest in time T, the one that
minimises the integral of squared jerk (third derivative of position) is

    s(tau) = 10 tau^3 - 15 tau^4 + 6 tau^5,      tau = t / T in [0, 1]

It starts and ends with zero velocity AND zero acceleration. Human reaching
movements follow this profile closely, which is why it was proposed (Flash & Hogan, 1985).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ari.robot.transforms import axis_angle_to_rot, make_T, rot_to_axis_angle


def min_jerk(tau: np.ndarray | float) -> np.ndarray | float:
    """Time-scaling s(tau): 0 -> 1 with zero velocity and acceleration at both ends."""
    tau = np.clip(tau, 0.0, 1.0)
    return tau**3 * (10 - 15 * tau + 6 * tau**2)


def min_jerk_velocity(tau: np.ndarray | float, duration: float) -> np.ndarray | float:
    """ds/dt for the min-jerk profile (in 1/s)."""
    tau = np.clip(tau, 0.0, 1.0)
    return 30 * tau**2 * (1 - tau) ** 2 / duration


@dataclass
class PoseTrajectory:
    """Straight-line position + single-axis rotation from T0 to T1, min-jerk timed."""

    T0: np.ndarray
    T1: np.ndarray
    duration: float

    def __post_init__(self):
        self._dR = rot_to_axis_angle(self.T0[:3, :3].T @ self.T1[:3, :3])

    def __call__(self, t: float) -> np.ndarray:
        """Desired 4x4 pose at time t (seconds since the trajectory started)."""
        s = min_jerk(t / self.duration) if self.duration > 0 else 1.0
        p = (1 - s) * self.T0[:3, 3] + s * self.T1[:3, 3]
        R = self.T0[:3, :3] @ axis_angle_to_rot(s * self._dR)
        return make_T(R, p)

    def done(self, t: float) -> bool:
        return t >= self.duration


def duration_for(T0: np.ndarray, T1: np.ndarray, max_speed: float = 0.25, max_rot_speed: float = 1.0,
                 min_duration: float = 0.3) -> float:
    """Pick a duration so the PEAK speed stays under the limits.

    Min-jerk peak velocity is 1.875 x the average velocity (at tau = 0.5), so for
    distance d we need T >= 1.875 d / v_max. Same for rotation angle.
    """
    d = np.linalg.norm(T1[:3, 3] - T0[:3, 3])
    ang = np.linalg.norm(rot_to_axis_angle(T0[:3, :3].T @ T1[:3, :3]))
    return max(min_duration, 1.875 * d / max_speed, 1.875 * ang / max_rot_speed)


@dataclass
class JointPathTrajectory:
    """Executes a list of joint-space waypoints, min-jerk timed on each segment.

    The arm comes to rest at every waypoint. That is wasteful but simple and safe:
    the robot is exactly on the collision-checked straight segments at all times.
    Smoother alternatives (splines through the waypoints) can cut corners that were
    never collision-checked, so they need re-checking; noted as future work.
    """

    waypoints: list
    max_joint_speed: float = 0.8  # rad/s peak, per joint

    def __post_init__(self):
        self.waypoints = [np.asarray(q, float) for q in self.waypoints]
        durs = [max(0.3, 1.875 * np.abs(b - a).max() / self.max_joint_speed)
                for a, b in zip(self.waypoints[:-1], self.waypoints[1:])]
        self.seg_durations = durs
        self.seg_starts = np.concatenate([[0.0], np.cumsum(durs)])
        self.duration = float(self.seg_starts[-1])

    def __call__(self, t: float) -> np.ndarray:
        if t >= self.duration or len(self.waypoints) == 1:
            return self.waypoints[-1].copy()
        i = int(np.searchsorted(self.seg_starts, t, side="right") - 1)
        s = min_jerk((t - self.seg_starts[i]) / self.seg_durations[i])
        return (1 - s) * self.waypoints[i] + s * self.waypoints[i + 1]

    def done(self, t: float) -> bool:
        return t >= self.duration


@dataclass
class DenseJointPath:
    """A finely sampled joint path (e.g. from a Cartesian straight line), traversed with
    ONE min-jerk time scaling over its whole length: smooth, no stops between points.

    points[i] corresponds to path parameter s_i = i / (n - 1), and the min-jerk profile
    maps time -> s, exactly like PoseTrajectory does for Cartesian poses.
    """

    points: list
    duration: float

    def __post_init__(self):
        self.points = np.asarray(self.points, float)

    def __call__(self, t: float) -> np.ndarray:
        s = min_jerk(t / self.duration) if self.duration > 0 else 1.0
        x = s * (len(self.points) - 1)
        i = min(int(x), len(self.points) - 2)
        f = x - i
        return (1 - f) * self.points[i] + f * self.points[i + 1]

    def done(self, t: float) -> bool:
        return t >= self.duration
