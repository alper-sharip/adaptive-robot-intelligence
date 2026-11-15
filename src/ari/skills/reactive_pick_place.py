"""M13/M14: a state-feedback ("reactive") pick-and-place expert.

The Stage 2 scripted policy is a clock-driven trajectory: what it does depends on how much
time has passed. This expert has NO memory and NO clock. Each step it looks at the current
state and decides: action = f(state). Two consequences:

  * It is learnable by a memoryless network: one state always maps to one action.
    (Behaviour cloning of the clock-driven expert averaged conflicting actions and stalled.)
  * It reacts. If the cube moves, slips, or is dropped, the next decision starts from where
    things are now. It can also label ANY state with the right action, which DAgger needs.

Decision rules:
  holding the cube      -> steer the cube over the target; height follows a FUNNEL: carry height
                           when far, sinking continuously to the place height as the horizontal
                           error goes to zero; open the gripper once down and centred
  cube already at target-> open, rise to carry height (done)
  otherwise             -> move over the cube and align yaw; height follows the same kind of
                           funnel down onto the cube; close the gripper once down and centred

Why funnels? The first version used hard thresholds ("within 3 mm: lower; otherwise: go back
up to carry height"). The expert itself never crossed them, but a learner that drifted 4 mm
while lowering met a rule demanding a 13 cm climb, and the cloned network, unable to fit a
cliff, stalled between the two answers (devlog.md). With funnels, a small error asks for a
small correction: the expert's action is a CONTINUOUS function of the state, which a network
can fit and which pulls nearby states back in.

Motion: aim at a point `max_step` ahead toward the goal (or the goal itself when closer).
The joint servos follow a command with a ~0.1 s lag, so a 25 mm lead gives ~0.25 m/s, and
the final approach decays smoothly without any explicit trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ari.learning.spaces import make_obs, wrap


@dataclass
class ReactiveExpert:
    approach_height: float = 0.12   # TCP height above the table while positioning over the cube
    carry_height: float = 0.15
    place_clearance: float = 0.005
    max_step: float = 0.025         # m lead per decision  (~0.25 m/s)
    max_yaw_step: float = 0.15      # rad lead per decision
    xy_tol: float = 0.003           # funnel bottom: horizontal error treated as zero (m)
    yaw_tol: float = 0.05           # rad
    z_tol: float = 0.004
    funnel_slope: float = 6.0       # m of height per m of horizontal error
    goal_radius: float = 0.02       # cube this close to the target counts as placed

    half: float = 0.02              # cube half-size (m)

    def reset(self) -> None:        # nothing to reset: no memory
        pass

    def act(self, env) -> np.ndarray:
        return self.act_from_obs(make_obs(env))

    __call__ = act

    def act_from_obs(self, obs: np.ndarray) -> np.ndarray:
        """The expert as a pure function of the observation vector (learning/spaces.py).
        Only the primary entries are read: gripper pose, finger opening, cube pose, target."""
        p = np.asarray(obs[0:3], dtype=float)
        yaw, fw = float(np.arctan2(obs[3], obs[4])), float(obs[5])
        cube = np.asarray(obs[6:9], dtype=float)
        cube_grasp_yaw = float(np.arctan2(obs[9], obs[10])) / 4.0     # in (-45, 45] deg
        tgt = np.asarray(obs[11:13], dtype=float)
        half, size = self.half, 2 * self.half

        near_hand = np.linalg.norm(cube - p) < 0.02
        holding = near_hand and fw < size + 0.004          # fingers have closed onto the cube
        cube_at_goal = np.linalg.norm(cube[:2] - tgt[:2]) < self.goal_radius and cube[2] < half + 0.012

        goal, goal_yaw, closed = p.copy(), yaw, False
        if holding:
            closed = True
            place_z = half + self.place_clearance
            exy = np.linalg.norm(tgt[:2] - cube[:2])       # steer the CUBE onto the target
            goal[2] = min(self.carry_height, place_z + self.funnel_slope * max(0.0, exy - self.xy_tol))
            # Sideways motion is faded in with height (never drag the cube along the table),
            # except close to the target, where small corrections are always allowed.
            gate = max(np.clip((p[2] - 0.05) / 0.05, 0, 1), np.clip((0.03 - exy) / 0.01, 0, 1))
            goal[:2] = p[:2] + gate * (tgt[:2] - cube[:2])
            if exy < 2 * self.xy_tol and p[2] < place_z + self.z_tol:
                closed = False                             # release
        elif cube_at_goal:
            goal[2] = self.carry_height                    # task done: back away upward
        else:
            goal_yaw = cube_grasp_yaw                      # in [-45, 45] deg: the wrist never winds up
            exy = np.linalg.norm(cube[:2] - p[:2])
            eyaw = abs(wrap(goal_yaw - yaw))
            err = max(0.0, exy - self.xy_tol) + 0.03 * max(0.0, eyaw - self.yaw_tol)
            goal[:2] = cube[:2]
            goal[2] = min(self.approach_height, cube[2] + self.funnel_slope * err)
            if exy < 2 * self.xy_tol and eyaw < 2 * self.yaw_tol and p[2] < cube[2] + self.z_tol:
                closed = True                              # grasp

        d = goal - p
        n = np.linalg.norm(d)
        if n > self.max_step:
            d *= self.max_step / n
        dyaw = float(np.clip(wrap(goal_yaw - yaw), -self.max_yaw_step, self.max_yaw_step))
        return np.array([*d, dyaw, float(closed)], dtype=np.float32)


def cliff_score(expert: ReactiveExpert, obs: np.ndarray, eps: float = 0.0015) -> float:
    """How discontinuous is the expert at this state? Largest change in its motion command (m)
    when the gripper position, the cube position or the finger opening is nudged by `eps`
    (1.5 mm); infinity if the gripper command flips.

    Smooth regions score ~eps. A score of centimetres means the state sits on a decision
    boundary: a 1.5 mm difference in what is observed flips what the expert wants.
    """
    base = expert.act_from_obs(obs)
    worst = 0.0
    for i in (0, 1, 2, 5, 6, 7, 8):
        for sign in (-1.0, 1.0):
            o = np.array(obs, dtype=np.float64)
            o[i] += sign * eps
            a = expert.act_from_obs(o)
            if a[4] != base[4]:
                return float("inf")
            worst = max(worst, float(np.linalg.norm(a[:3] - base[:3])))
    return worst
