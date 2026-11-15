"""Loaded by pytest before any test module.

Importing `ari` first lets it choose MuJoCo's headless renderer on display-less Linux
(see src/ari/__init__.py); a test file that imports `mujoco` before `ari` would be too late.
"""

import ari

ARI_VERSION = ari.__version__  # referenced so linters keep the import
