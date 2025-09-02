#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Franka Panda robot interface with direct connection management."""

import time
import traceback
from typing import Any, Dict, Optional, Tuple

import numpy as np

from maniunicon.robot_interface.base import RobotInterface
from maniunicon.utils.ik_solver import IKSolver
from maniunicon.utils.shared_memory.shared_storage import RobotAction, RobotState

try:
    import franka_py
    FRANKA_AVAILABLE = True
except ImportError:
    FRANKA_AVAILABLE = False
    print("franka_py not installed. Please install it to use Franka Panda interface.")


class FrankaPandaInterface(RobotInterface):
    """Franka Panda robot interface with direct connection management."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Franka Panda robot interface.

        Args:
            config: Dictionary containing robot configuration parameters
        """
        super().__init__(config)
        
        # Extract Franka-specific config
        self.ip = config.get("ip", "172.16.0.2")
        self.velocity = config.get("velocity", 0.5)
        self.acceleration = config.get("acceleration", 0.5)
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
        if not FRANKA_AVAILABLE:
            print("franka_py is not installed. Cannot connect to Franka Panda.")
            return False
            
        try:
            # Connect to Franka Panda robot
            self.robot = franka_py.FrankaArm(self.ip)
            
            # Connect to gripper
            try:
                self.gripper = franka_py.FrankaGripper(self.ip)
                self.gripper.open()
                self._gripper_state = np.array([0.0])
            except Exception as e:
                print(f"Warning: Could not initialize gripper: {e}")
                self.gripper = None
            
            # Initialize IK solver
            self.ik_solver = IKSolver(self.config)
            
            # Set default collision behavior
            self.robot.set_collision_behavior(
                lower_torque_thresholds_nominal=[20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
                upper_torque_thresholds_nominal=[20.0, 20.0, 18.0, 18.0, 16.0, 14.0, 12.0],
                lower_force_thresholds_nominal=[10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
                upper_force_thresholds_nominal=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
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
            if self.robot is not None:
                self.robot.close()
                self.robot = None
                
            if self.gripper is not None:
                self.gripper.close()
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
                
                # Move to initial position
                self.robot.move_to_joint_positions(init_qpos)
                
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
                self.robot.move_to_joint_positions(joint_positions.tolist())
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
            # Get robot state
            state = self.robot.get_robot_state()
            
            # Extract joint information
            joint_positions = np.array(state.q)
            joint_velocities = np.array(state.dq)
            joint_torques = np.array(state.tau_J)
            
            # Get TCP pose
            tcp_position, tcp_orientation = self.forward_kinematics(joint_positions)
            
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
                    # Position control
                    self.robot.move_to_joint_positions(
                        action.joint_positions.tolist(),
                        time_to_go=self.dt
                    )
                elif action.joint_velocities is not None:
                    # Velocity control
                    self.robot.set_joint_velocities(action.joint_velocities.tolist())
                elif action.joint_torques is not None:
                    # Torque control
                    self.robot.set_joint_torques(action.joint_torques.tolist())
                    
            elif action.control_mode == "cartesian":
                # Cartesian control
                if (action.tcp_position is not None and 
                    action.tcp_orientation is not None):
                    
                    # Get current joint positions
                    state = self.robot.get_robot_state()
                    current_q = np.array(state.q)
                    
                    # Solve IK
                    joint_solution = self.inverse_kinematics(
                        action.tcp_position,
                        action.tcp_orientation,
                        current_q
                    )
                    
                    if joint_solution is not None:
                        # Send joint command
                        self.robot.move_to_joint_positions(
                            joint_solution.tolist(),
                            time_to_go=self.dt
                        )
                    else:
                        print("IK solution failed")
                        return False
            
            # Handle gripper control
            if action.gripper_state is not None and self.gripper is not None:
                if action.gripper_state.item():  # action is close
                    if not self._gripper_state.item():  # current is open
                        print("Closing gripper")
                        self.gripper.grasp(width=0.0, force=40.0)
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
                state = self.robot.get_robot_state()
                # Check for errors in robot state
                return state.robot_mode != franka_py.RobotMode.kMove
            return self._error_state
        except:
            return True
    
    def clear_error(self) -> bool:
        """Clear any error state."""
        if not self._is_connected:
            return False
            
        try:
            if self.robot is not None:
                self.robot.automatic_error_recovery()
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
            if self.ik_solver is not None:
                return self.ik_solver.get_tcp_pose(joint_positions)
            else:
                # Fallback: use robot's built-in FK if available
                if self.robot is not None:
                    pose = self.robot.forward_kinematics(joint_positions.tolist())
                    # Convert pose matrix to position and quaternion
                    position = pose[:3, 3]
                    # Convert rotation matrix to quaternion
                    from scipy.spatial.transform import Rotation
                    rotation = Rotation.from_matrix(pose[:3, :3])
                    quaternion = rotation.as_quat()  # [x, y, z, w]
                    return position, quaternion
                else:
                    raise RuntimeError("No FK solver available")
                    
        except Exception as e:
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
            if self.ik_solver is not None:
                # Use generic IK solver
                joint_solution, success = self.ik_solver.solve(
                    target_position=target_position,
                    target_orientation=target_orientation,
                    dt=self.dt,
                    current_q=current_q,
                )
                return joint_solution if success else None
            else:
                # Fallback: use robot's built-in IK if available
                if self.robot is not None and hasattr(self.robot, 'inverse_kinematics'):
                    # Convert quaternion to rotation matrix
                    from scipy.spatial.transform import Rotation
                    rotation = Rotation.from_quat(target_orientation)
                    rotation_matrix = rotation.as_matrix()
                    
                    # Create 4x4 transformation matrix
                    pose = np.eye(4)
                    pose[:3, :3] = rotation_matrix
                    pose[:3, 3] = target_position
                    
                    # Solve IK
                    joint_solution = self.robot.inverse_kinematics(pose, current_q.tolist())
                    return np.array(joint_solution) if joint_solution is not None else None
                else:
                    print("No IK solver available")
                    return None
                    
        except Exception as e:
            print(f"Error in inverse kinematics: {e}")
            self._error_state = True
            return None