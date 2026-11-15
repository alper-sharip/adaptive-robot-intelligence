"""Sensor noise, so simulated perception isn't unrealistically perfect.

Depth: structured-light / stereo depth cameras have error that grows roughly with the
square of distance, plus missing pixels (no return at edges, shiny or dark surfaces).
    sigma(z) = a + b * z^2
'realistic' is in the range quoted for consumer RGB-D cameras at ~1 m (a few mm);
'poor' is roughly 3x worse. These are modelling assumptions, not a specific sensor's datasheet.

Colour: per-pixel Gaussian noise plus a global brightness change (lighting variation).
Missing depth pixels are returned as 0, which is what real drivers do; perception must
ignore them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class NoiseModel:
    depth_a: float = 0.0          # m
    depth_b: float = 0.0          # 1/m
    depth_dropout: float = 0.0    # fraction of pixels with no depth
    rgb_sigma: float = 0.0        # intensity units (0-255)
    brightness: float = 0.0       # max relative global brightness change, e.g. 0.2 = +/-20%

    def apply(self, rgb: np.ndarray, depth: np.ndarray, rng: np.random.Generator):
        depth = depth.astype(np.float64)
        if self.depth_a or self.depth_b:
            depth = depth + rng.normal(0.0, 1.0, depth.shape) * (self.depth_a + self.depth_b * depth**2)
        if self.depth_dropout:
            depth[rng.random(depth.shape) < self.depth_dropout] = 0.0
        rgb = rgb.astype(np.float64)
        if self.brightness:
            rgb *= 1.0 + rng.uniform(-self.brightness, self.brightness)
        if self.rgb_sigma:
            rgb += rng.normal(0.0, self.rgb_sigma, rgb.shape)
        return np.clip(rgb, 0, 255).astype(np.uint8), depth


NOISE_LEVELS = {
    "none": NoiseModel(),
    "realistic": NoiseModel(depth_a=0.0005, depth_b=0.0025, depth_dropout=0.02, rgb_sigma=6, brightness=0.15),
    "poor": NoiseModel(depth_a=0.0015, depth_b=0.0075, depth_dropout=0.10, rgb_sigma=15, brightness=0.30),
}
