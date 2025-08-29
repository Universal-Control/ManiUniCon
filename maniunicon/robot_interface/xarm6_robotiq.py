#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""XArm robot interface with direct connection management."""

import time
import traceback
from typing import Any, Dict

import numpy as np
import pinocchio as pin
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI
from pyRobotiqGripper import RobotiqGripper

from maniunicon.robot_interface.base import RobotInterface
from maniunicon.utils.ik_solver import IKSolver
from maniunicon.utils.shared_memory.shared_storage import RobotAction, RobotState


class XArm6RobotiqInterface(RobotInterface):
    """XArm robot plus Robotiq gripper interface with direct connection management."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the XArm robot interface.

        Args:
            config: Dictionary containing robot configuration parameters
        """
        super().__init__(config)

        # Extract XArm-specific config
        self.ip = config.get("ip", "192.168.124.243")
        self.velocity = config.get("velocity", 0.5)
        self.acceleration = config.get("acceleration", 0.5)
        self.dt = config.get("dt", 1.0 / 30.0)  # Default 30Hz control rate

        # Robot connections
        self.arm = None
        self.ik_solver = None
        self._current_state = None
        self._is_connected = False
        self._error_state = False
        self.gripper_state = np.array([0.0])  # Gripper open state

    def connect(self) -> bool:
        """Connect to the XArm robot."""
        try:
            # Connect to XArm robot
            self.arm = XArmAPI(self.ip)

            # Initialize robot
            self.arm.motion_enable(enable=True)
            self.arm.clean_error()
            self.arm.set_mode(1)  # Position control mode
            self.arm.set_state(0)  # Sport state

            # Initialize Robotiq gripper
            try:
                self.gripper = RobotiqGripper()
                self.gripper.activate()
                self.gripper.open()
                self.gripper_state = np.array([0.0])  # Gripper open state
                print("Robotiq gripper initialized successfully")
            except Exception as e:
                print(f"Error initializing Robotiq gripper: {e}")
                traceback.print_exc()
                self.gripper = None

            # Initialize IK solver
            self.ik_solver = IKSolver(self.config)

            self._is_connected = True
            self._error_state = False
            print(f"Successfully connected to XArm robot at {self.ip}")
            return True

        except Exception as e:
            print(f"Error connecting to XArm robot: {e}")
            traceback.print_exc()
            self._is_connected = False
            self._error_state = True
            return False

    def disconnect(self) -> bool:
        """Disconnect from the XArm robot."""
        try:
            if self.arm is not None:
                self.arm.disconnect()
                self.arm = None

            self.ik_solver = None
            self._is_connected = False
            print("Disconnected from XArm robot")
            return True

        except Exception as e:
            print(f"Error during XArm disconnect: {e}")
            return False

    def reset_to_init(self) -> bool:
        """Reset the robot to the initial configuration."""
        if self.arm is not None:
            try:
                self.arm.set_mode(0)
                self.arm.set_state(0)
                # Move to initial joint position
                self.arm.set_servo_angle(
                    angle=self.config["init_qpos"],
                    wait=True,
                    is_radian=True,
                )
                # Open Robotiq gripper
                if self.gripper is not None:
                    self.gripper.open()
                    self.gripper_state = np.array([0.0])  # Gripper open state

                print("XArm reset to initial position")
                self.arm.set_mode(1)
                self.arm.set_state(0)
                return True

            except Exception as e:
                print(f"Error resetting XArm: {e}")
                return False
        return False

    def move_to_joint_positions(self, joint_positions: np.ndarray) -> bool:
        if self.arm is not None:
            try:
                print(f"moving to joint positions {joint_positions}")
                self.arm.set_mode(0)
                self.arm.set_state(0)
                # Move to initial joint position
                self.arm.set_servo_angle(
                    angle=joint_positions,
                    wait=True,
                    is_radian=True,
                )

                self.arm.set_mode(1)
                self.arm.set_state(0)
                print(f"moved to joint positions {joint_positions}")
                return True
            except Exception as e:
                print(f"Error moving joints: {e}")
                return False
        return False

    def get_state(self) -> RobotState:
        """Get the current state of the XArm robot."""
        if not self.is_connected():
            raise RuntimeError("Robot is not connected")

        try:
            # Get joint state
            joint_positions = np.array(self.arm.get_servo_angle()[1])[
                :6
            ]  # [1] contains the angles
            joint_positions = np.deg2rad(joint_positions)  # Convert to radians

            # Get joint velocities (XArm API might not provide this directly)
            try:
                joint_velocities = np.array(self.arm.get_joint_speed()[1])
                joint_velocities = np.deg2rad(joint_velocities)  # Convert to rad/s
            except:
                joint_velocities = np.zeros_like(joint_positions)

            # Get joint torques (XArm API might not provide this directly)
            try:
                joint_torques = np.array(self.arm.get_joint_torque()[1])
            except:
                joint_torques = np.zeros_like(joint_positions)

            # Get TCP pose using IK solver
            tcp_position, tcp_orientation = self.ik_solver.get_tcp_pose(joint_positions)

            # Get Robotiq gripper state
            if self.gripper is not None:
                gripper_state = self.gripper_state
            else:
                # No gripper available
                gripper_state = np.array([0.0])

            current_state = RobotState(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_torques=joint_torques,
                tcp_position=tcp_position,
                tcp_orientation=tcp_orientation,
                gripper_state=gripper_state,
                timestamp=np.array(time.time()),
            )

            self._current_state = current_state
            return current_state

        except Exception as e:
            print(f"Error getting robot state: {e}")
            traceback.print_exc()
            self._error_state = True
            if self._current_state is None:
                raise RuntimeError("No robot state available")
            return self._current_state

    def send_action(self, action: RobotAction) -> bool:
        """Send a control action to the XArm robot."""
        if not self.is_connected():
            return False

        if not self.validate_action(action):
            print("Invalid action: exceeds robot limits")
            return False

        try:
            if action.control_mode == "joint":
                # Direct joint control
                if action.joint_positions is not None:
                    self.arm.set_servo_angle_j(
                        angles=action.joint_positions.tolist(),
                        speed=0.5,
                        is_radian=True,
                    )
                elif action.joint_velocities is not None:
                    # XArm doesn't directly support velocity control in servo mode
                    # This would require implementing velocity control logic
                    print("Joint velocity control not implemented for XArm")
                    return False

            elif action.control_mode == "cartesian":
                # Cartesian control
                if (
                    action.tcp_position is not None
                    and action.tcp_orientation is not None
                ):
                    # Get current joint positions for IK
                    current_q = np.array(self.arm.get_servo_angle()[1])[:6]
                    current_q = np.deg2rad(current_q)

                    # Solve IK
                    joint_solution, success = self.ik_solver.solve(
                        target_position=np.array(action.tcp_position),
                        target_orientation=np.array(action.tcp_orientation),
                        dt=self.dt,
                        current_q=current_q,
                    )

                    if success:
                        # Convert to degrees and send joint command
                        self.arm.set_servo_angle_j(
                            angles=joint_solution.tolist(),
                            speed=0.5,
                            is_radian=True,
                        )
                    else:
                        print("IK solution failed")
                        return False

            # Handle Robotiq gripper control
            if action.gripper_state is not None and self.gripper is not None:
                if action.gripper_state.item():  # action is close
                    if not self.gripper_state.item():  # current is open
                        print("Closing Robotiq gripper")
                        self.gripper_state = action.gripper_state
                        self.gripper.close()
                else:  # action is open
                    if self.gripper_state.item():  # current is close
                        print("Opening Robotiq gripper")
                        self.gripper_state = action.gripper_state
                        self.gripper.open()

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

        try:
            # Check XArm error state
            if self.arm is not None:
                state = self.arm.get_state()
                # State codes: 1=in motion, 2=sleeping, 3=paused, 4=error, 5=go_stop, 6=emergency_stop
                if state[1] in [4, 5, 6]:
                    return True
            return self._error_state
        except:
            return True

    def clear_error(self) -> bool:
        """Clear any error state."""
        if not self._is_connected:
            return False

        try:
            self.arm.clean_error()
            self.arm.clean_warn()
            self.arm.set_mode(0)
            self.arm.set_state(state=0)
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
            self.arm.emergency_stop()
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
