#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Franka Panda robot interface using franky library."""

import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from maniunicon.robot_interface.base import RobotInterface
from maniunicon.utils.ik_solver import IKSolver
from maniunicon.utils.shared_memory.shared_storage import RobotAction, RobotState

try:
    from franky import Affine, CartesianMotion, JointPositions, Motion, Robot, Gripper
    from franky import RobotState as FrankyRobotState
    from franky import Measure
    FRANKY_AVAILABLE = True
except ImportError:
    FRANKY_AVAILABLE = False
    print("franky not installed. Please install it with 'pip install franky-panda'")


class FrankaPandaInterface(RobotInterface):
    """Franka Panda robot interface using franky library."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Franka Panda robot interface.

        Args:
            config: Dictionary containing robot configuration parameters
        """
        super().__init__(config)
        
        # Extract Franka-specific config
        self.ip = config.get("ip", "172.16.0.12")
        self.velocity_rel = config.get("velocity", 0.2)  # Relative velocity [0-1]
        self.acceleration_rel = config.get("acceleration", 0.1)  # Relative acceleration [0-1]
        self.jerk_rel = config.get("jerk", 0.01)  # Relative jerk [0-1]
        self.dt = config.get("dt", 1.0 / 1000.0)  # Default 1kHz control rate
        
        # Franka Panda has 7 joints
        self.num_joints = 7
        
        # Joint limits for Franka Panda (in radians)
        self.joint_limits = config.get("joint_limits", {
            "min": np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]),
            "max": np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        })
        
        # Velocity limits (rad/s)
        self.velocity_limits = config.get("velocity_limits", 
            np.array([2.175, 2.175, 2.175, 2.175, 2.610, 2.610, 2.610]))
        
        # Torque limits (Nm)
        self.torque_limits = config.get("torque_limits",
            np.array([87, 87, 87, 87, 12, 12, 12]))
        
        # Robot connections
        self.robot = None
        self.gripper = None
        self.ik_solver = None
        self._current_state = None
        self._is_connected = False
        self._error_state = False
        self._gripper_state = np.array([0.0])  # 0: open, 1: closed
        
    def connect(self) -> bool:
        """Connect to the Franka Panda robot."""
        if not FRANKY_AVAILABLE:
            print("franky is not installed. Cannot connect to Franka Panda.")
            return False
            
        try:
            # Connect to Franka Panda robot using franky
            self.robot = Robot(self.ip)
            self.robot.relative_dynamics_factor = 0.2  # Safety factor for dynamics
            
            # Set default velocity, acceleration and jerk
            self.robot.velocity_rel = self.velocity_rel
            self.robot.acceleration_rel = self.acceleration_rel
            self.robot.jerk_rel = self.jerk_rel
            
            # Connect to gripper
            try:
                self.gripper = Gripper(self.ip)
                self.gripper.max_width = 0.08  # 80mm max width
                self.gripper.open()
                self._gripper_state = np.array([0.0])
            except Exception as e:
                print(f"Warning: Could not initialize gripper: {e}")
                self.gripper = None
            
            # Initialize IK solver
            self.ik_solver = IKSolver(self.config)
            
            # Set collision thresholds for safety
            self.robot.set_collision_behavior(
                [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],  # lower_torque_thresholds
                [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],  # upper_torque_thresholds
                [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],  # lower_torque_thresholds_acceleration
                [20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],  # upper_torque_thresholds_acceleration
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  # lower_force_thresholds
                [20.0, 20.0, 20.0, 20.0, 20.0, 20.0],  # upper_force_thresholds
                [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],  # lower_force_thresholds_acceleration
                [20.0, 20.0, 20.0, 20.0, 20.0, 20.0]   # upper_force_thresholds_acceleration
            )
            
            self._is_connected = True
            self._error_state = False
            print(f"Successfully connected to Franka Panda robot at {self.ip}")
            return True
            
        except Exception as e:
            print(f"Error connecting to Franka Panda robot: {e}")
            traceback.print_exc()
            self._is_connected = False
            self._error_state = True
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from the Franka Panda robot."""
        try:
            # franky handles disconnection automatically
            self.robot = None
            self.gripper = None
            self.ik_solver = None
            self._is_connected = False
            print("Disconnected from Franka Panda robot")
            return True
            
        except Exception as e:
            print(f"Error during Franka Panda disconnect: {e}")
            return False
    
    def reset_to_init(self) -> bool:
        """Reset the robot to the initial configuration."""
        if self.robot is not None:
            try:
                print("Resetting to initial position...")
                
                # Default initial joint positions for Franka Panda
                init_qpos = self.config.get("init_qpos", 
                    [0, -0.785, 0, -2.356, 0, 1.571, 0.785])
                
                # Create joint motion
                motion = JointPositions(init_qpos)
                
                # Move to initial position
                self.robot.move(motion)
                
                # Open gripper
                if self.gripper is not None:
                    self.gripper.open()
                    self._gripper_state = np.array([0.0])
                
                print("Reset finished!")
                return True
            except Exception as e:
                print(f"Error during reset: {e}")
                return False
        return False
    
    def move_to_joint_positions(self, joint_positions: np.ndarray) -> bool:
        """Move robot to specified joint positions."""
        if self.robot is not None:
            try:
                print(f"Moving to joint positions {joint_positions}")
                motion = JointPositions(joint_positions.tolist())
                self.robot.move(motion)
                print(f"Moved to joint positions {joint_positions}")
                return True
            except Exception as e:
                print(f"Error moving to joint positions: {e}")
                return False
        return False
    
    def get_state(self) -> RobotState:
        """Get the current state of the Franka Panda robot."""
        if not self.is_connected():
            raise RuntimeError("Robot is not connected")
            
        try:
            # Get robot state from franky
            state = self.robot.state
            
            # Extract joint information
            joint_positions = np.array(state.q)
            joint_velocities = np.array(state.dq)
            joint_torques = np.array(state.tau_J)
            
            # Get TCP pose (franky provides O_T_EE as 4x4 matrix)
            ee_pose = np.array(state.O_T_EE).reshape(4, 4)
            tcp_position = ee_pose[:3, 3]
            
            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_matrix(ee_pose[:3, :3])
            tcp_orientation = rotation.as_quat()  # [x, y, z, w]
            
            # Get gripper state
            if self.gripper is not None:
                gripper_state = self._gripper_state
            else:
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
            self._error_state = True
            if self._current_state is None:
                raise RuntimeError("No robot state available")
            return self._current_state
    
    def send_action(self, action: RobotAction) -> bool:
        """Send a control action to the Franka Panda robot."""
        if not self.is_connected():
            return False
            
        if not self.validate_action(action):
            print("Invalid action: exceeds robot limits")
            return False
            
        try:
            if action.control_mode == "joint":
                # Direct joint control
                if action.joint_positions is not None:
                    # Position control using franky
                    motion = JointPositions(action.joint_positions.tolist())
                    # For continuous control, use async move
                    self.robot.move_async(motion, asynchronous=True)
                    
                elif action.joint_velocities is not None:
                    # Velocity control - franky doesn't have direct velocity control
                    # We can approximate it using position control with small timesteps
                    current_q = np.array(self.robot.state.q)
                    target_q = current_q + action.joint_velocities * self.dt
                    
                    # Clip to joint limits
                    target_q = np.clip(target_q, self.joint_limits["min"], self.joint_limits["max"])
                    
                    motion = JointPositions(target_q.tolist())
                    self.robot.move_async(motion, asynchronous=True)
                    
                elif action.joint_torques is not None:
                    # Torque control - franky doesn't expose direct torque control
                    # This would require libfranka's torque interface
                    print("Direct torque control not supported in franky interface")
                    return False
                    
            elif action.control_mode == "cartesian":
                # Cartesian control
                if (action.tcp_position is not None and 
                    action.tcp_orientation is not None):
                    
                    # Convert quaternion to rotation matrix
                    from scipy.spatial.transform import Rotation
                    rotation = Rotation.from_quat(action.tcp_orientation)
                    rotation_matrix = rotation.as_matrix()
                    
                    # Create affine transformation
                    affine = Affine(action.tcp_position.tolist(), rotation_matrix.tolist())
                    
                    # Create Cartesian motion
                    motion = CartesianMotion(affine)
                    
                    # Execute motion
                    self.robot.move_async(motion, asynchronous=True)
            
            # Handle gripper control
            if action.gripper_state is not None and self.gripper is not None:
                if action.gripper_state.item():  # action is close
                    if not self._gripper_state.item():  # current is open
                        print("Closing gripper")
                        self.gripper.grasp(0.01, 40.0)  # 10mm width, 40N force
                        self._gripper_state = action.gripper_state
                else:  # action is open
                    if self._gripper_state.item():  # current is closed
                        print("Opening gripper")
                        self.gripper.open()
                        self._gripper_state = action.gripper_state
            
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
            if self.robot is not None:
                # Check if robot has errors
                return self.robot.has_errors()
            return self._error_state
        except:
            return True
    
    def clear_error(self) -> bool:
        """Clear any error state."""
        if not self._is_connected:
            return False
            
        try:
            if self.robot is not None:
                self.robot.recover_from_errors()
                self._error_state = False
                return True
            return False
        except Exception as e:
            print(f"Error clearing robot error: {e}")
            return False
    
    def stop(self) -> bool:
        """Emergency stop the robot."""
        if not self._is_connected:
            return False
            
        try:
            if self.robot is not None:
                self.robot.stop()
                return True
            return False
        except Exception as e:
            print(f"Error stopping robot: {e}")
            return False
    
    def forward_kinematics(
        self, joint_positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
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
            # Use franky's kinematics
            kinematics = self.robot.get_kinematics()
            ee_pose = kinematics.forward(joint_positions.tolist())
            
            # Extract position
            tcp_position = np.array([ee_pose.translation.x, 
                                     ee_pose.translation.y, 
                                     ee_pose.translation.z])
            
            # Convert rotation to quaternion
            from scipy.spatial.transform import Rotation
            rotation_matrix = np.array(ee_pose.rotation).reshape(3, 3)
            rotation = Rotation.from_matrix(rotation_matrix)
            tcp_orientation = rotation.as_quat()  # [x, y, z, w]
            
            return tcp_position, tcp_orientation
                    
        except Exception as e:
            # Fallback to IK solver if available
            if self.ik_solver is not None:
                return self.ik_solver.get_tcp_pose(joint_positions)
            else:
                print(f"Error in forward kinematics: {e}")
                self._error_state = True
                raise
    
    def inverse_kinematics(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        current_q: np.ndarray,
    ) -> Optional[np.ndarray]:
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
            # Convert quaternion to rotation matrix
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_quat(target_orientation)
            rotation_matrix = rotation.as_matrix()
            
            # Create affine transformation
            affine = Affine(target_position.tolist(), rotation_matrix.tolist())
            
            # Use franky's inverse kinematics
            kinematics = self.robot.get_kinematics()
            joint_solution = kinematics.inverse(affine.matrix(), current_q.tolist())
            
            if joint_solution is not None:
                return np.array(joint_solution)
            else:
                # Fallback to IK solver if franky IK fails
                if self.ik_solver is not None:
                    joint_solution, success = self.ik_solver.solve(
                        target_position=target_position,
                        target_orientation=target_orientation,
                        dt=self.dt,
                        current_q=current_q,
                    )
                    return joint_solution if success else None
                else:
                    print("IK solution not found")
                    return None
                    
        except Exception as e:
            print(f"Error in inverse kinematics: {e}")
            self._error_state = True
            return None