"""Adaptive Robot Intelligence: autonomous manipulation in simulation."""

import os
import sys

# Headless Linux (servers, Colab, CI) has no display for MuJoCo's default OpenGL backend.
# Pick the GPU-offscreen backend automatically unless the user chose one. Must run before
# `mujoco` is imported anywhere, which is why it lives in the package's __init__.
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    os.environ.setdefault("MUJOCO_GL", "egl")

__version__ = "0.1.0"
