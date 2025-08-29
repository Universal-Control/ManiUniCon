from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Tuple

from maniunicon.utils.shared_memory.shared_storage import RobotAction, RobotState


class RobotInterface(ABC):
    """Base class for robot interfaces."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.num_joints = config.get("num_joints", 6)
        self.joint_limits = config.get("joint_limits", {})
        self.velocity_limits = config.get("velocity_limits", {})
        self.torque_limits = config.get("torque_limits", {})

    @abstractmethod
    def connect(self) -> bool:
        """Connect to the robot hardware."""
        pass

    @abstractmethod
    def disconnect(self) -> bool:
        """Disconnect from the robot hardware."""
        pass

    @abstractmethod
    def get_state(self) -> RobotState:
        """Get the current state of the robot."""
        pass

    @abstractmethod
    def send_action(self, action: RobotAction) -> bool:
        """Send a control action to the robot."""
        pass

    @abstractmethod
    def reset_to_init(self) -> bool:
        """Reset the robot to the initial configuration."""
        pass

    @abstractmethod
    def forward_kinematics(
        self, joint_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the forward kinematics of the robot."""
        pass

    @abstractmethod
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        current_q: np.ndarray,
    ) -> np.ndarray:
        """Compute the inverse kinematics of the robot.

        Args:
            target_position: Target TCP position [x, y, z]
            target_orientation: Target TCP orientation as quaternion [x, y, z, w]
            current_q: Current joint positions to use as IK seed

        Returns:
            Joint positions if successful, None if failed
        """
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the robot is connected."""
        pass

    @abstractmethod
    def is_error(self) -> bool:
        """Check if the robot is in an error state."""
        pass

    @abstractmethod
    def clear_error(self) -> bool:
        """Clear any error state."""
        pass

    @abstractmethod
    def stop(self) -> bool:
        """Emergency stop the robot."""
        pass

    def validate_action(self, action: RobotAction) -> bool:
        """Validate if an action is within robot limits."""
        if action.control_mode == "joint":
            if action.joint_positions is not None:
                for i, pos in enumerate(action.joint_positions):
                    if not (
                        self.joint_limits["min"][i]
                        <= pos
                        <= self.joint_limits["max"][i]
                    ):
                        return False

            elif action.joint_velocities is not None:
                for i, vel in enumerate(action.joint_velocities):
                    if not (abs(vel) <= self.velocity_limits[i]):
                        return False

            elif action.joint_torques is not None:
                for i, torque in enumerate(action.joint_torques):
                    if not (abs(torque) <= self.torque_limits[i]):
                        return False

        return True
