"""Robot interface modules for ManiUniCon."""

from maniunicon.robot_interface import dummy, meshcat

try:
    from maniunicon.robot_interface import ur5_robotiq
except ImportError:
    print(
        "UR5 robot interface not installed. Please install it with `pip install ur_rtde`"
    )

try:
    from maniunicon.robot_interface import franka_panda_deoxys
except ImportError:
    print(
        "Franka Panda deoxys interface not installed. Please install deoxys_control"
    )


try:
    from maniunicon.robot_interface import franka_fr3_franky
except ImportError:
    print(
        "Franka FR3 Franky interface not installed. Please install franka_ros"
    )


__all__ = [
    "dummy",
    "meshcat",
    "ur5_robotiq",
    "franka_panda_deoxys",
    "franka_fr3_franky",
]
