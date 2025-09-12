"""ManiUniCon - A unified control interface for robotic manipulation."""

from pathlib import Path
from maniunicon import core, robot_interface, utils

PROJECT_ROOT = Path(__file__).parent.parent

__version__ = "0.1.0"
__all__ = [
    "core",
    "robot_interface",
    "utils",
]
