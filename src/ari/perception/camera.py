"""M10: a pinhole RGB-D camera: intrinsics, extrinsics, projection and back-projection.

Conventions (the source of most camera bugs, so they are spelled out):

* OpenCV camera frame, used for all maths here: x right, y DOWN, z FORWARD (into the scene).
* MuJoCo / OpenGL camera frame: x right, y UP, z BACKWARD. The camera looks along its -z.
  Converting between them flips y and z: R_cv = R_gl @ diag(1, -1, -1).
* Pixels: column u to the right, row v downward, origin at the image's top-left CORNER.
  Pixel (row r, col c) covers [c, c+1) x [r, r+1); its centre is (u, v) = (c + 0.5, r + 0.5).

Pinhole model (intrinsics K):
    u = fx * X/Z + cx,   v = fy * Y/Z + cy        (X, Y, Z in the OpenCV camera frame)
MuJoCo specifies a camera by its vertical field of view, fovy, so
    fy = (H/2) / tan(fovy/2),  fx = fy (square pixels),  cx = W/2,  cy = H/2.

Depth: MuJoCo's depth image is the distance along the optical axis (Z), not the
Euclidean distance to the point. Back-projection therefore uses Z directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from ari.robot.transforms import make_T

GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """MuJoCo camera quaternion (w, x, y, z) for a camera at `pos` looking at `target`."""
    pos, target, up = (np.asarray(a, float) for a in (pos, target, up))
    back = pos - target                       # MuJoCo camera z-axis points BACKWARD
    back /= np.linalg.norm(back)
    right = np.cross(up, back)
    right /= np.linalg.norm(right)
    cam_up = np.cross(back, right)
    R = np.column_stack([right, cam_up, back])
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, R.reshape(-1))
    return q


@dataclass
class CameraModel:
    K: np.ndarray              # 3x3 intrinsics
    T_world_cam: np.ndarray    # 4x4 pose of the OpenCV camera frame in the world
    width: int
    height: int

    @classmethod
    def from_mujoco(cls, model: mujoco.MjModel, data: mujoco.MjData, name: str, width: int, height: int):
        cam = model.camera(name).id
        fovy = np.deg2rad(model.cam_fovy[cam])
        fy = (height / 2) / np.tan(fovy / 2)
        K = np.array([[fy, 0, width / 2], [0, fy, height / 2], [0, 0, 1.0]])
        R_gl = data.cam_xmat[cam].reshape(3, 3)
        T = make_T(R_gl @ GL_TO_CV, data.cam_xpos[cam].copy())
        return cls(K, T, width, height)

    def project(self, p_world: np.ndarray):
        """World points (N,3) -> pixel coords (N,2) [u, v] and depth Z (N,)."""
        p = np.atleast_2d(p_world)
        T_cw = np.linalg.inv(self.T_world_cam)
        pc = (T_cw[:3, :3] @ p.T).T + T_cw[:3, 3]
        uv = (self.K @ (pc / pc[:, 2:3]).T).T[:, :2]
        return uv, pc[:, 2]

    def backproject(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Pixel coords + depth -> world points (N,3)."""
        u, v, z = (np.asarray(a, float).ravel() for a in (u, v, z))
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        pc = np.column_stack([(u - cx) * z / fx, (v - cy) * z / fy, z])
        return (self.T_world_cam[:3, :3] @ pc.T).T + self.T_world_cam[:3, 3]

    def backproject_pixels(self, rows: np.ndarray, cols: np.ndarray, depth_img: np.ndarray) -> np.ndarray:
        """Back-project pixel CENTRES (row, col) using the depth image."""
        return self.backproject(cols + 0.5, rows + 0.5, depth_img[rows, cols])


class RGBDCamera:
    """Renders colour + depth (+ optional ground-truth segmentation for grading only)."""

    def __init__(self, model: mujoco.MjModel, name: str = "front_cam", width: int = 640, height: int = 480):
        self.model, self.name, self.width, self.height = model, name, width, height
        self.rgb_renderer = mujoco.Renderer(model, height=height, width=width)
        self.depth_renderer = mujoco.Renderer(model, height=height, width=width)
        self.depth_renderer.enable_depth_rendering()
        self.seg_renderer = None

    def capture(self, data: mujoco.MjData, with_gt_segmentation: bool = False):
        """Returns (rgb uint8 HxWx3, depth float32 HxW in metres, camera model, gt_seg or None)."""
        self.rgb_renderer.update_scene(data, camera=self.name)
        rgb = self.rgb_renderer.render()
        self.depth_renderer.update_scene(data, camera=self.name)
        depth = self.depth_renderer.render()
        seg = None
        if with_gt_segmentation:
            if self.seg_renderer is None:
                self.seg_renderer = mujoco.Renderer(self.model, height=self.height, width=self.width)
                self.seg_renderer.enable_segmentation_rendering()
            self.seg_renderer.update_scene(data, camera=self.name)
            seg = self.seg_renderer.render()  # (H, W, 2): object id, object type
        cam = CameraModel.from_mujoco(self.model, data, self.name, self.width, self.height)
        return rgb, depth, cam, seg

    def close(self):
        for r in (self.rgb_renderer, self.depth_renderer, self.seg_renderer):
            if r is not None:
                r.close()
