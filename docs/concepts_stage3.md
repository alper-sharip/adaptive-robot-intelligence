# Concept notes: Stage 3 perception

## 1. The pinhole camera: `perception/camera.py`
A 3-D point (X, Y, Z) in the camera frame lands on pixel
`u = fx·X/Z + cx`, `v = fy·Y/Z + cy`. The 3×3 matrix K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
holds the **intrinsics**. MuJoCo defines a camera by its vertical field of view, so
`fy = (H/2) / tan(fovy/2)`. The camera's pose in the world, T_world_cam, is the **extrinsics**.

Projection loses Z (depth). An RGB-D camera measures it, so the mapping can be inverted:
`X = (u − cx)·Z / fx`, `Y = (v − cy)·Z / fy`. That's **back-projection**: every pixel with depth
becomes a 3-D point.

## 2. Conventions: where most camera bugs live
- OpenCV camera frame: x right, **y down, z forward**. OpenGL/MuJoCo: x right, **y up, z backward**.
  Converting flips two axes: `R_cv = R_gl · diag(1, −1, −1)`.
- Pixel (row r, col c) has its centre at (u, v) = (c + 0.5, r + 0.5).
- MuJoCo's depth is the distance along the optical axis (Z), not the straight-line distance.
The test `test_rendered_depth_matches_geometry` checks all three at once against the renderer.

## 3. Segmentation by colour: `perception/detect.py`
HSV separates *what colour* (hue) from *how bright* (value), so shading changes V but barely H.
Red sits at both ends of the hue circle (0 and 179), hence two ranges. Thresholds were set from
measured pixel statistics, not guessed.

## 4. From pixels to an object pose
Mask → back-project → drop points at table height (removes the floor reflection) → find the top
face (mode of heights, by mean-shift) → centroid for position, minimum-area rectangle for yaw.
Yaw is folded to [−45°, 45°) because a cube looks the same every 90°.

## 5. Noise and bias
- **Noise** spreads estimates out; averaging many pixels shrinks it (√N).
- **Bias** shifts every estimate the same way; averaging does nothing. It comes from the method,
  e.g. selecting points by a noisy value (the "highest points" bug).
- A median filter reduces per-pixel noise and fills dropped pixels while keeping edges.
- Evaluate with noise you can control: `perception/noise.py` (σ(z) = a + b·z², dropouts,
  colour noise, brightness changes).

## 6. Grading perception honestly
The simulator can render a perfect segmentation. It's used only in tests and evaluation, to
grade the colour-based detector, and never to drive the robot. Errors are measured against simulator ground truth for
every episode (`evaluate_vision.py`).

## Check your understanding
1. If you forgot the half-pixel offset (used u = c instead of c + 0.5), roughly how large an xy
   error would that cause at 0.9 m from a camera with fy ≈ 580 px?
2. Why does depth noise along the viewing ray create an *xy* error when the camera is tilted?
3. The target error is ~1 mm even with no noise, while the cube's is ~0.3 mm. Propose a cause and a test.
4. Which pieces of information does the vision-driven robot still get from the simulator? Which
   would be hardest to replace with perception?
