#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""UR5 robot interface with direct connection management."""

import time
import traceback
from typing import Any, Dict

import numpy as np
from franky import (
    Affine,
    Robot,
    Gripper,
    JointMotion,
    JointVelocityMotion,
    RelativeDynamicsFactor,
)

from maniunicon.robot_interface.base import RobotInterface
from maniunicon.utils.ik_solver import IKSolver
from maniunicon.utils.shared_memory.shared_storage import RobotAction, RobotState


class FRANKAInterface(RobotInterface):
    """FRANKA robot plus parallel gripper interface with direct connection management."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the FRANKA robot interface.

        Args:
            config: Dictionary containing robot configuration parameters
        """
        super().__init__(config)

        # Extract FRANKA-specific config
        self.ip = config.get("ip", "172.16.0.2")
        self.lookahead_time = config.get("lookahead_time", 0.1)
        self.gain = config.get("gain", 300)
        self.dt = config.get("dt", 1.0 / 125.0)  # Default 125Hz control rate
        self.gripper_control_speed = config.get("gripper_control_speed", 0.1)
        self.gripper_control_force = config.get("gripper_control_force", 20)

        # Robot connections
        self.robot = None
        self.ik_solver = None
        self._current_state = None
        self._is_connected = False
        self._error_state = False
        self._last_time_get_action = None

    def connect(self) -> bool:
        """Connect to the UR5 robot."""
        try:
            # Connect to UR5 robot
            self.robot = Robot(self.ip)
            self.robot.recover_from_errors()
            self.robot.relative_dynamics_factor = RelativeDynamicsFactor(
                velocity=0.3, acceleration=0.3, jerk=0.05
            )

            try:
                self.gripper = Gripper(self.ip)

                self.gripper.open(self.gripper_control_speed)
                self.gripper_state = np.array([1.0])  # Gripper open state
            except Exception as e:
                print(f"Error initializing gripper: {e}")
                traceback.print_exc()
                self.gripper = None

            # Initialize IK solver
            self.ik_solver = IKSolver(self.config)

            self._is_connected = True
            self._error_state = False
            print(f"Successfully connected to UR5 robot at {self.ip}")
            return True

        except Exception as e:
            print(f"Error connecting to UR5 robot: {e}")
            traceback.print_exc()
            self._is_connected = False
            self._error_state = True
            return False

    def disconnect(self) -> bool:
        """Disconnect from the UR5 robot."""
        try:
            if self.robot is not None:
                del self.robot
                del self.gripper
                self.robot = None
                self.gripper = None

            self.ik_solver = None
            self._is_connected = False
            print("Disconnected from UR5 robot")
            return True

        except Exception as e:
            print(f"Error during UR5 disconnect: {e}")
            return False

    def reset_to_init(self) -> bool:
        """Reset the robot to the initial configuration."""
        if self.robot is not None:
            print("init!")
            self.robot.move(JointMotion(self.config["init_qpos"]))
            time.sleep(1)
            self.gripper.open(self.gripper_control_speed)
            self.gripper_state = np.array([0.0])  # Gripper open state
            print("init finished!")
            return True
        return False

    def move_to_joint_positions(self, joint_positions: np.ndarray) -> bool:
        if self.robot is not None:
            print(f"moving to joint positions {joint_positions}")
            self.robot.move(JointMotion(joint_positions))
            print(f"moved to joint positions {joint_positions}")
            return True
        return False

    def get_state(self) -> RobotState:
        """Get the current state of the UR5 robot."""
        if not self.is_connected():
            raise RuntimeError("Robot is not connected")

        try:
            state = self.robot.state
            # Get joint state
            joint_positions = state.q
            joint_velocities = state.dq
            joint_torques = state.tau_J

            # Get TCP pose using IK solver
            tcp_position, tcp_orientation = self.ik_solver.get_tcp_pose(
                np.array(joint_positions)
            )

            if self.gripper is None:
                gripper_state = np.array([0.0])  # placeholder here
            else:
                # 0 for open, 1 for closed
                gripper_state = self.gripper_state
                # TODO(zbzhu): add read/write lock of gripper channel
                # TODO(zbzhu): check the value of `100` here
                # gripper_state = np.array(
                #     (self.gripper.getPosition() < 100)
                # ).astype(np.float32)
            current_state = RobotState(
                joint_positions=np.array(joint_positions),
                joint_velocities=np.array(joint_velocities),
                joint_torques=np.array(joint_torques),
                tcp_position=tcp_position,
                tcp_orientation=tcp_orientation,
                gripper_state=gripper_state,
                timestamp=np.array(time.time()),
            )

            self._current_state = current_state
            return current_state

        except Exception as e:
            print(f"Error getting robot state: {e}")
            self._error_state = True
            if self._current_state is None:
                raise RuntimeError("No robot state available")
            return self._current_state

    def send_action(self, action: RobotAction) -> bool:
        """Send a control action to the UR5 robot."""
        if self._last_time_get_action is None:
            self._last_time_get_action = time.time()
        else:
            print(
                f"Time since last get action: {(time.time() - self._last_time_get_action) * 1000} ms"
            )
            self._last_time_get_action = time.time()

        if not self.is_connected():
            return False

        if not self.validate_action(action):
            print("Invalid action: exceeds robot limits")
            return False

        try:
            if action.control_mode == "joint":
                # Direct joint control
                if action.joint_positions is not None:
                    action.joint_positions = self._clip_joint_positions(
                        action.joint_positions
                    )
                    motion = JointMotion(action.joint_positions)
                    self.robot.move(motion, asynchronous=True)
                elif action.joint_velocities is not None:
                    motion = JointVelocityMotion(action.joint_velocities)
                    self.robot.move(motion, asynchronous=True)
                elif action.joint_torques is not None:
                    raise NotImplementedError("Torque control is not supported")

            elif action.control_mode == "cartesian":
                raise NotImplementedError("Cartesian control is not supported")

            if action.gripper_state is not None and self.gripper is not None:
                if action.gripper_state.item():  # action is close
                    if not self.gripper_state.item():  # current is open
                        print("Closing gripper")
                        self.gripper_state = action.gripper_state
                        self.gripper.grasp(
                            0.0,
                            self.gripper_control_speed,
                            self.gripper_control_force,
                            epsilon_outer=1.0,
                        )
                else:  # action is open
                    if self.gripper_state:  # current is close
                        print("Opening gripper")
                        self.gripper_state = action.gripper_state
                        self.gripper.open(self.gripper_control_speed)

            # regulate frequency
            # NOTE(zbzhu): currently we use outside control loop to regulate frequency
            # self.robot_c.waitPeriod(t_start)

            return True

        except Exception as e:
            print(f"Error sending action: {e}")
            traceback.print_exc()
            self._error_state = True
            return False

    def is_connected(self) -> bool:
        """Check if the robot is connected."""
        return self._is_connected and not self._error_state

    def is_error(self) -> bool:
        """Check if the robot is in an error state."""
        if not self._is_connected:
            return True

        # Check for robot-specific errors
        try:
            # You can add specific UR5 error checking here
            # For example, check robot mode, safety status, etc.
            return self._error_state
        except:
            return True

    def clear_error(self) -> bool:
        """Clear any error state."""
        if not self._is_connected:
            return False

        try:
            self.robot_c.stopScript()
            self.robot_c.unlockProtectiveStop()
            self._error_state = False
            return True
        except Exception as e:
            print(f"Error clearing robot error: {e}")
            return False

    def stop(self) -> bool:
        """Emergency stop the robot."""
        if not self._is_connected:
            return False

        try:
            return True
        except Exception as e:
            print(f"Error stopping robot: {e}")
            return False

    def forward_kinematics(
        self, joint_positions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute forward kinematics to get TCP pose from joint positions.

        Args:
            joint_positions: Array of joint positions in radians

        Returns:
            tuple: (tcp_position, tcp_orientation) where:
                - tcp_position is a 3D array [x, y, z] in meters
                - tcp_orientation is a quaternion [x, y, z, w]
        """
        if not self.is_connected():
            raise RuntimeError("Robot is not connected")

        try:
            # Update robot configuration
            q = np.array(joint_positions)
            return self.ik_solver.get_tcp_pose(q)

        except Exception as e:
            print(f"Error in forward kinematics: {e}")
            self._error_state = True
            raise

    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        current_q: np.ndarray,
    ) -> np.ndarray:
        """Compute inverse kinematics to get joint positions from TCP pose.

        Args:
            target_position: Target TCP position [x, y, z] in meters
            target_orientation: Target TCP orientation as quaternion [x, y, z, w]
            current_q: Current joint positions to use as IK seed

        Returns:
            Joint positions if successful, None if failed
        """
        if not self.is_connected():
            raise RuntimeError("Robot is not connected")

        try:
            # Solve IK
            joint_solution, success = self.ik_solver.solve(
                target_position=target_position,
                target_orientation=target_orientation,
                dt=self.dt,
                current_q=current_q,
            )

            return joint_solution if success else None

        except Exception as e:
            print(f"Error in inverse kinematics: {e}")
            self._error_state = True
            return None

    def _clip_joint_positions(
        self,
        joint_positions: np.ndarray,
        max_threshold: float = 0.6,
        min_threshold: float = 0.01,
    ) -> np.ndarray:
        """Clip joint positions, ensure each step is within the limits."""

        state = self.get_state()
        current_joint_positions = state.joint_positions
        delta_joint_positions = joint_positions - current_joint_positions
        if np.max(np.abs(delta_joint_positions)) > max_threshold:
            print(
                f"FRANKAInterface: delta_joint_positions {delta_joint_positions} exceeds threshold {max_threshold}, clipped, max {np.max(np.abs(delta_joint_positions))}"
            )
        delta_joint_positions = np.clip(
            delta_joint_positions, -max_threshold, max_threshold
        )
        delta_joint_positions = np.where(
            np.abs(delta_joint_positions) < min_threshold, 0, delta_joint_positions
        )
        return current_joint_positions + delta_joint_positions


if __name__ == "__main__":
    joint = [0.0, 0.0, 0.0, -2.2, 0.0, 2.2, 0.7]
    robot = Robot("172.16.0.2")
    robot.recover_from_errors()
    robot.relative_dynamics_factor = 0.1
    robot.move(JointMotion(joint))
    time.sleep(1)
    robot.move(JointMotion(joint))
