"""Kinematics of the SO-101 arm, written independently of MuJoCo and checked against it.

The SO-101 (TheRobotStudio / Hugging Face) has five arm joints plus a gripper:
    1 shoulder_pan  (vertical axis)     2 shoulder_lift, 3 elbow_flex, 4 wrist_flex (all parallel)
    5 wrist_roll    (along the forearm)  + gripper (one moving jaw)

Forward kinematics: the arm description gives, for each link, a fixed transform from its
parent and a joint that turns about the link's own z-axis. FK is the product
    T = F_1 Rz(q1) F_2 Rz(q2) F_3 Rz(q3) F_4 Rz(q4) F_5 Rz(q5) F_tool
The fixed transforms F_i are read from the arm description (the MuJoCo Menagerie model),
the way a DH table is read from a datasheet; the chaining is done here.

Inverse kinematics, in closed form: unlike the 7-joint Panda, this arm has exactly as many
joints as a top-down grasp has constraints (position 3, pointing down 1, yaw 1), and its
geometry makes each angle solvable by formula:
  - joints 2-4 are parallel, so after the base turns, the arm is a planar two-link chain;
  - the wrist-roll axis passes through the wrist-flex axis, so the wrist is a fixed point;
  - the tool point sits a fixed distance below the wrist and 12 mm off the roll axis.
`ik_top_down` derives every angle from those facts (see its docstring for the equations),
and the tests check it against FK on random targets.

All geometric constants are measured from the description at load time (`Geometry`), not
typed in, so a different calibration of the same arm needs no code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ari.robot.transforms import make_T, quat_to_rot, rot_z

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
TOOL_SITE = "gripperframe"


def menagerie_path() -> Path:
    import os

    root = Path(os.environ.get("ARI_MENAGERIE", Path(__file__).resolve().parents[3].parent / "mujoco_menagerie"))
    return root / "robotstudio_so101"


@dataclass
class Chain:
    fixed: list            # F_1 .. F_5: parent -> link frame at q = 0 (4x4)
    tool: np.ndarray       # wrist_roll link -> tool site (4x4)
    base: np.ndarray       # world -> arm base body (4x4)
    limits: np.ndarray     # (5, 2)


def load_chain(model) -> Chain:
    """Read the fixed transforms of the five arm links from a MuJoCo model of the SO-101."""
    import mujoco

    fixed, limits = [], []
    for name in JOINTS:
        j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        b = model.jnt_bodyid[j]
        assert np.allclose(model.jnt_axis[j], [0, 0, 1]) and np.allclose(model.jnt_pos[j], 0), name
        fixed.append(make_T(quat_to_rot(model.body_quat[b]), model.body_pos[b].copy()))
        limits.append(model.jnt_range[j].copy())
    s = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, TOOL_SITE)
    tool = make_T(quat_to_rot(model.site_quat[s]), model.site_pos[s].copy())
    base_body = model.body_parentid[model.jnt_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, JOINTS[0])]]
    base = make_T(quat_to_rot(model.body_quat[base_body]), model.body_pos[base_body].copy())
    return Chain(fixed, tool, base, np.array(limits))


def fk_frames(chain: Chain, q: np.ndarray) -> list:
    """World poses of the five joint frames (after each joint's rotation), then the tool."""
    T = chain.base.copy()
    out = []
    for F, qi in zip(chain.fixed, q):
        T = T @ F @ make_T(rot_z(qi), np.zeros(3))
        out.append(T.copy())
    out.append(T @ chain.tool)
    return out


def fk(chain: Chain, q: np.ndarray) -> np.ndarray:
    return fk_frames(chain, q)[-1]


# ---------------------------------------------------------------- closed-form IK
@dataclass
class Geometry:
    """The numbers the closed-form IK needs, all measured from the chain (`measure_geometry`)."""
    pan_xy: np.ndarray       # where the vertical pan axis meets the table plane
    pan_sign: float          # world rotation = Rz(pan_sign * q1)
    radial0: np.ndarray      # outward horizontal direction of the arm plane at q1 = 0
    normal0: np.ndarray      # horizontal normal of the arm plane at q1 = 0 (= pitch axis direction)
    lateral: float           # the arm plane's sideways offset from the pan axis (along normal0)
    shoulder: np.ndarray     # shoulder-lift axis in the arm plane: (radius from pan axis, height)
    L1: float                # shoulder -> elbow
    L2: float                # elbow -> wrist
    phi1: float              # upper-arm direction at q = 0, in the arm plane
    phi2: float              # forearm direction at q = 0
    phi_tool: float          # tool direction (wrist -> tool along the roll axis) at q = 0
    pitch_sign: float        # in-plane angles change by pitch_sign * q for joints 2-4
    tool_len: float          # wrist point -> tool point, along the roll axis
    tool_off: float          # tool point's distance from the roll axis
    off_angle: float         # direction of that offset relative to the tool yaw (top-down tool)
    yaw0: float              # top-down tool: yaw = yaw0 + a1*q1 + a5*q5
    a1: float
    a5: float


def tool_yaw(T: np.ndarray) -> float:
    """Yaw of a (near) top-down tool: direction of the tool frame's y-axis in the table plane."""
    return float(np.arctan2(T[1, 1], T[0, 1]))


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _angle(v) -> float:
    return float(np.arctan2(v[1], v[0]))


def measure_geometry(chain: Chain) -> Geometry:
    Fs = fk_frames(chain, np.zeros(5))
    p_pan, z_pan = Fs[0][:3, 3], Fs[0][:3, 2]
    assert abs(abs(z_pan[2]) - 1) < 1e-9, "pan axis must be vertical"
    n = Fs[1][:3, 2]
    assert all(abs(abs(Fs[i][:3, 2] @ n) - 1) < 1e-9 for i in (2, 3)), "joints 2-4 must be parallel"
    assert abs(n[2]) < 1e-9, "pitch axes must be horizontal"
    radial = np.cross(n, [0, 0, 1.0])
    if radial @ (Fs[3][:3, 3] - p_pan) < 0:
        radial = -radial

    def plane(p):
        v = p - p_pan
        return np.array([v @ radial, p[2]])

    P2, P3, P4 = (plane(Fs[i][:3, 3]) for i in (1, 2, 3))
    roll_p, roll_a = Fs[4][:3, 3], Fs[4][:3, 2]
    wrist = Fs[3][:3, 3] + ((roll_p - Fs[3][:3, 3]) @ n) * n      # point of the wrist axis on the roll axis
    assert np.linalg.norm(np.cross(roll_a, wrist - roll_p)) < 1e-6, "roll axis must meet the wrist axis"
    tool = Fs[5][:3, 3]
    if roll_a @ (tool - wrist) < 0:
        roll_a = -roll_a
    along = (tool - wrist) @ roll_a
    off = tool - wrist - along * roll_a
    e = 1e-6
    dq = np.zeros(5)
    dq[1] = e
    d_ang = _angle(plane(fk_frames(chain, dq)[2][:3, 3]) - P2) - _angle(P3 - P2)
    g = Geometry(pan_xy=p_pan[:2].copy(), pan_sign=float(np.sign(z_pan[2])), radial0=radial, normal0=n,
                 lateral=float((wrist - p_pan) @ n), shoulder=P2,
                 L1=float(np.linalg.norm(P3 - P2)), L2=float(np.linalg.norm(P4 - P3)),
                 phi1=_angle(P3 - P2), phi2=_angle(P4 - P3),
                 phi_tool=_angle(np.array([roll_a @ radial, roll_a[2]])),
                 pitch_sign=float(np.sign(d_ang)), tool_len=float(along), tool_off=float(np.linalg.norm(off)),
                 off_angle=0.0, yaw0=0.0, a1=0.0, a5=0.0)
    # yaw and tool-offset direction, measured in a top-down reference pose
    q2, q3 = _planar(g, np.array([0.22, 0.20]), True)
    q_ref = np.array([0.0, q2, q3, _q4_down(g, q2, q3), 0.0])
    Fr = fk_frames(chain, q_ref)
    assert abs(Fr[5][2, 0] ** 2 + Fr[5][2, 1] ** 2 + Fr[5][2, 2] ** 2 - 1) < 1e-9
    g.yaw0 = tool_yaw(Fr[5])
    g.a1 = float(np.round((tool_yaw(fk(chain, q_ref + [e, 0, 0, 0, 0])) - g.yaw0) / e))
    g.a5 = float(np.round((tool_yaw(fk(chain, q_ref + [0, 0, 0, 0, e])) - g.yaw0) / e))
    v = Fr[5][:3, 3] - Fr[4][:3, 3]
    ra = Fr[4][:3, 2]
    o = v - (v @ ra) * ra
    g.off_angle = _angle(o) - g.yaw0 if g.tool_off > 1e-9 else 0.0
    return g


def _planar(g: Geometry, wrist_rz: np.ndarray, elbow_up: bool) -> tuple[float, float]:
    """Shoulder and elbow angles that put the wrist at (radius, height) in the arm plane.

    Link directions:  a1 = phi1 + s q2,  a2 = phi2 + s (q2 + q3).
    Law of cosines:   |D|^2 = L1^2 + L2^2 + 2 L1 L2 cos(a2 - a1),  D = wrist - shoulder.
    """
    D = wrist_rz - g.shoulder
    c = (D @ D - g.L1**2 - g.L2**2) / (2 * g.L1 * g.L2)
    if abs(c) > 1:
        raise ValueError(f"out of reach (shoulder-wrist {1000 * np.linalg.norm(D):.1f} mm, "
                         f"arm spans {1000 * abs(g.L1 - g.L2):.1f}-{1000 * (g.L1 + g.L2):.1f} mm)")
    gam = np.arccos(c) * (1 if elbow_up else -1)
    a1 = _angle(D) - np.arctan2(g.L2 * np.sin(gam), g.L1 + g.L2 * np.cos(gam))
    s = g.pitch_sign
    q2 = (a1 - g.phi1) / s
    q3 = ((a1 + gam) - g.phi2) / s - q2
    return float(_wrap(q2)), float(_wrap(q3))


def _q4_down(g: Geometry, q2: float, q3: float) -> float:
    """Tool direction = phi_tool + s (q2 + q3 + q4); straight down is -90 deg."""
    return float(_wrap((-np.pi / 2 - g.phi_tool) / g.pitch_sign - q2 - q3))


@dataclass
class IKSolution:
    q: np.ndarray
    ok: bool
    reason: str = ""


def ik_top_down(chain: Chain, g: Geometry, p: np.ndarray, yaw: float, elbow_up: bool | None = None) -> IKSolution:
    """Both elbow solutions are computed; the first one inside the joint limits is returned
    (elbow_up=True/False forces one). See `_ik_branch` for the equations."""
    branches = (elbow_up,) if elbow_up is not None else (False, True)
    sols = [_ik_branch(chain, g, p, yaw, b) for b in branches]
    return next((s for s in sols if s.ok), sols[0])


def _ik_branch(chain: Chain, g: Geometry, p: np.ndarray, yaw: float, elbow_up: bool) -> IKSolution:
    """Closed-form IK: tool point at p (world), pointing straight down, turned to `yaw`.

      1. roll axis:   c = p - tool_off * (cos(yaw + off_angle), sin(yaw + off_angle), 0)
      2. wrist:       w = c + (0, 0, tool_len)                  (the tool hangs below it)
      3. base:        r = sqrt(|w_xy - pan|^2 - e^2)
                      s1 q1 = atan2(w - pan) - atan2(r * radial0 + e * normal0)
      4. shoulder, elbow: two-link planar IK to (r, w_z)     (law of cosines, `_planar`)
      5. wrist flex:  q4 = (-pi/2 - phi_tool) / s - q2 - q3   (tool straight down)
      6. wrist roll:  q5 = (yaw - yaw0 - a1 q1) / a5
    Every quantity is known before it is used, so nothing is iterative.
    """
    p = np.asarray(p, dtype=float)
    a = yaw + g.off_angle
    w = p - g.tool_off * np.array([np.cos(a), np.sin(a), 0.0]) + np.array([0.0, 0.0, g.tool_len])
    uv = w[:2] - g.pan_xy
    rr = uv @ uv - g.lateral**2
    if rr <= 0:
        return IKSolution(np.zeros(5), False, "target on the pan axis")
    r = np.sqrt(rr)
    arm0 = r * g.radial0[:2] + g.lateral * g.normal0[:2]
    q1 = _wrap(_angle(uv) - _angle(arm0)) / g.pan_sign
    try:
        q2, q3 = _planar(g, np.array([r, w[2]]), elbow_up)
    except ValueError as err:
        return IKSolution(np.zeros(5), False, str(err))
    q4 = _q4_down(g, q2, q3)
    q5 = _wrap((yaw - g.yaw0 - g.a1 * q1) / g.a5)
    q = np.array([q1, q2, q3, q4, q5])
    lo, hi = chain.limits[:, 0], chain.limits[:, 1]
    for k in (-1, 1):                                   # the roll joint's range exceeds one turn slightly
        if not (lo[4] <= q[4] <= hi[4]) and lo[4] <= q[4] + 2 * np.pi * k <= hi[4]:
            q[4] += 2 * np.pi * k
    bad = [JOINTS[i] for i in range(5) if not (lo[i] - 1e-9 <= q[i] <= hi[i] + 1e-9)]
    return IKSolution(q, not bad, f"joint limits: {', '.join(bad)}" if bad else "")
