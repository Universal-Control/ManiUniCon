import time
import numpy as np

from maniunicon.robot_interface.base import RobotInterface
from maniunicon.utils.shared_memory.shared_storage import RobotAction, RobotState


class DummyRobot(RobotInterface):
    def connect(self) -> bool:
        print("Dummy robot connected")
        return True

    def disconnect(self) -> bool:
        print("Dummy robot disconnected")
        return True

    def get_state(self) -> RobotState:
        return RobotState(
            joint_positions=np.zeros(6),
            joint_velocities=np.zeros(6),
            joint_torques=np.zeros(6),
            tcp_position=np.zeros(3),
            tcp_orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            gripper_state=np.array([0.0]),
            timestamp=np.array(time.time()),
        )

    def send_action(self, action: RobotAction) -> bool:
        print(f"Sending action: {action}")
        return True

    def is_connected(self) -> bool:
        return True

    def is_error(self) -> bool:
        return False

    def clear_error(self) -> bool:
        return True

    def stop(self) -> bool:
        print("Dummy robot stopped")
        return True

    def reset_to_init(self) -> bool:
        print("Dummy robot reset to init")
        return True

    def forward_kinematics(
        self, joint_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Dummy forward kinematics - returns dummy TCP pose."""
        return np.array([0.0, 0.0, 0.5]), np.array([0.0, 0.0, 0.0, 1.0])

    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        current_q: np.ndarray,
    ) -> np.ndarray:
        """Dummy inverse kinematics - returns dummy joint positions."""
        return np.zeros(6)
