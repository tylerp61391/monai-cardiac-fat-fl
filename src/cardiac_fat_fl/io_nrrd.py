"""Backward-compatibility module alias.

All universal I/O and MetaTensor functions have moved to `cardiac_fat_fl.io_metatensor`.
This file preserves backwards compatibility for legacy scripts and imports.
"""

from cardiac_fat_fl.io_metatensor import *  # noqa: F401, F403
