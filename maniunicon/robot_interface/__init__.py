"""Robot interface modules for ManiUniCon."""

from maniunicon.robot_interface import dummy, meshcat

try:
    from maniunicon.robot_interface import ur5_robotiq
except ImportError:
    print(
        "UR5 robot interface not installed. Please install it with `pip install ur_rtde`"
    )


__all__ = [
    "dummy",
    "meshcat",
    "ur5_robotiq",
]
