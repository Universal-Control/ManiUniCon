"""Robot interface modules for ManiUniCon."""

from maniunicon.robot_interface import dummy, meshcat

try:
    from maniunicon.robot_interface import ur5_robotiq
except ImportError:
    print(
        "UR5 robot interface not installed. Please install it with `pip install ur_rtde`"
    )

try:
    from maniunicon.robot_interface import franka_panda
except ImportError:
    print(
        "Franka Panda robot interface not installed. Please install it with `pip install franka-py`"
    )


__all__ = [
    "dummy",
    "meshcat",
    "ur5_robotiq",
    "franka_panda",
]
