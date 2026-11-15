"""M11: find the cube and the target zone in an RGB-D image and localize them in 3-D.

Pipeline for the cube:
  1. Colour segmentation in HSV (hue is robust to shading; red wraps around hue 0).
     Thresholds come from measured pixel statistics (docs/results_stage3.md).
  2. Clean the mask (morphological opening), keep the largest connected component.
  3. Back-project the masked pixels to 3-D world points using depth + camera model.
  4. Drop points near the table (z < 5 mm). The floor is a mirror, and the cube's
     reflection is red too: colour alone sees a ghost cube; depth puts it at table height.
  5. The top face is the set of points within a few mm of the highest ones. Its centroid
     gives x, y; its height minus half the cube size gives z.
  6. Yaw from the minimum-area rectangle around the top-face points (cv2.minAreaRect),
     folded to [-45, 45) degrees because a cube repeats every 90 degrees.

The target is a flat green disc: segment, back-project, take the centroid.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ari.perception.camera import CameraModel


@dataclass
class ColorRange:
    """HSV thresholds in OpenCV units: H 0-179, S and V 0-255."""
    h_ranges: tuple          # ((lo, hi), ...) allows wrap-around for red
    s_min: int
    v_min: int


RED = ColorRange(h_ranges=((0, 10), (170, 179)), s_min=70, v_min=50)
GREEN = ColorRange(h_ranges=((60, 90),), s_min=90, v_min=60)


def color_mask(rgb: np.ndarray, cr: ColorRange) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = np.zeros(rgb.shape[:2], bool)
    for lo, hi in cr.h_ranges:
        mask |= (hsv[..., 0] >= lo) & (hsv[..., 0] <= hi)
    mask &= (hsv[..., 1] >= cr.s_min) & (hsv[..., 2] >= cr.v_min)
    return mask


def largest_component(mask: np.ndarray, open_px: int = 2) -> np.ndarray:
    m = mask.astype(np.uint8)
    if open_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * open_px + 1, 2 * open_px + 1))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    best = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == best


@dataclass
class CubeEstimate:
    ok: bool
    position: np.ndarray | None = None   # cube centre, world frame (m)
    yaw: float | None = None             # rad, folded to [-pi/4, pi/4)
    n_mask_px: int = 0
    n_top_points: int = 0
    reason: str = ""


@dataclass
class TargetEstimate:
    ok: bool
    position: np.ndarray | None = None
    n_mask_px: int = 0
    reason: str = ""


def fold_yaw(yaw: float) -> float:
    return (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4


def top_plane_height(z: np.ndarray, band: float, iters: int = 10) -> float:
    """Height of the flat top face: the MODE of the point heights, found by mean-shift.

    A flat top face concentrates its points at one height; side faces spread theirs out.
    Start at the fullest 1 mm histogram bin, then repeatedly move to the mean of the points
    within +/- band until it stops moving. The band is SYMMETRIC around the estimate.

    Two earlier versions, kept as lessons (docs/devlog.md):
      * 'highest points' (95th percentile minus a band) picks points whose noise pushed them
        up: +6.8 mm height bias with noisy depth, and xy bias through the oblique view;
      * a histogram smoothed over +/- band blurred the top-face spike into the side faces
        and gave 44 deg yaw errors even on PERFECT depth.
    """
    if z.max() - z.min() < 0.002:
        return float(np.mean(z))
    hist, edges = np.histogram(z, bins=np.arange(z.min(), z.max() + 0.001, 0.001))
    h = edges[int(np.argmax(hist))] + 0.0005
    for _ in range(iters):
        near = z[np.abs(z - h) <= band]
        if len(near) == 0:
            break
        h_new = float(near.mean())
        if abs(h_new - h) < 1e-5:
            break
        h = h_new
    return h


def localize_cube(rgb, depth, cam: CameraModel, half_size: float, *, color: ColorRange = RED,
                  min_px: int = 30, table_clearance: float = 0.005, top_band: float = 0.004,
                  depth_filter: int = 5, top_method: str = "plane") -> CubeEstimate:
    """top_method: 'plane' (histogram peak, symmetric band) or 'percentile' (first version, biased)."""
    if depth_filter:
        # Median filter: removes per-pixel noise and fills isolated dropouts, keeps edges sharp.
        depth = cv2.medianBlur(depth.astype(np.float32), depth_filter)
    mask = largest_component(color_mask(rgb, color))
    n = int(mask.sum())
    if n < min_px:
        return CubeEstimate(False, n_mask_px=n, reason="not_seen")
    r, c = np.nonzero(mask & (depth > 0))           # depth 0 = sensor returned nothing
    pts = cam.backproject_pixels(r, c, depth)
    pts = pts[pts[:, 2] > table_clearance]          # removes floor bleed and the mirror ghost
    if len(pts) < min_px // 2:
        return CubeEstimate(False, n_mask_px=n, reason="no_points_above_table")
    if top_method == "percentile":
        z_top = np.percentile(pts[:, 2], 95)
        top = pts[pts[:, 2] > z_top - top_band]
    else:
        z_top = top_plane_height(pts[:, 2], top_band)
        top = pts[np.abs(pts[:, 2] - z_top) <= top_band]
    if len(top) < 5:
        return CubeEstimate(False, n_mask_px=n, reason="top_face_too_small")
    xy = top[:, :2]
    rect = cv2.minAreaRect(xy.astype(np.float32) * 1000.0)   # work in mm: OpenCV prefers O(1) numbers
    (cx, cy), _, angle_deg = rect
    position = np.array([cx / 1000.0, cy / 1000.0, float(np.mean(top[:, 2])) - half_size])
    return CubeEstimate(True, position, fold_yaw(np.deg2rad(angle_deg)), n, len(top))


def localize_target(rgb, depth, cam: CameraModel, *, color: ColorRange = GREEN, min_px: int = 30,
                    depth_filter: int = 5) -> TargetEstimate:
    if depth_filter:
        depth = cv2.medianBlur(depth.astype(np.float32), depth_filter)
    mask = largest_component(color_mask(rgb, color))
    n = int(mask.sum())
    if n < min_px:
        return TargetEstimate(False, n_mask_px=n, reason="not_seen")
    r, c = np.nonzero(mask & (depth > 0))
    pts = cam.backproject_pixels(r, c, depth)
    pts = pts[np.abs(pts[:, 2]) < 0.03]  # the disc lies on the table (loose: depth noise)
    if len(pts) < min_px // 2:
        return TargetEstimate(False, n_mask_px=n, reason="no_table_points")
    return TargetEstimate(True, np.array([*np.median(pts[:, :2], axis=0), 0.0]), n)
