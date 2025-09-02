#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Franka Panda robot interface using deoxys_control library."""

import os
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from maniunicon.robot_interface.base import RobotInterface
from maniunicon.utils.ik_solver import IKSolver
from maniunicon.utils.shared_memory.shared_storage import RobotAction, RobotState

try:
    from deoxys import config_root
    from deoxys.franka_interface import FrankaInterface
    from deoxys.utils import YamlConfig
    from deoxys.utils.config_utils import get_default_controller_config
    from deoxys.utils.transform_utils import quat2mat, mat2quat, euler2mat, mat2euler
    DEOXYS_AVAILABLE = True
except ImportError:
    DEOXYS_AVAILABLE = False
    print("deoxys not installed. Please install deoxys_control to use this interface.")


class FrankaPandaDeoxysInterface(RobotInterface):
    """Franka Panda robot interface using deoxys_control library."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Franka Panda robot interface with deoxys.

        Args:
            config: Dictionary containing robot configuration parameters
        """
        super().__init__(config)
        
        # Extract deoxys-specific config
        self.deoxys_config_path = config.get("deoxys_config", None)
        assert self.deoxys_config_path is not None, "deoxys_config_path is not set"
        assert os.path.exists(self.deoxys_config_path), "deoxys_config_path does not exist"

        self.control_freq = config.get("control_freq", 20.0)  # Hz
        self.state_freq = config.get("state_freq", 100.0)  # Hz
        self.use_visualizer = config.get("use_visualizer", False)
        
        # Controller configuration
        self.default_controller_type = config.get("controller_type", "OSC_POSE")
        self.controller_config_file = config.get(
            "controller_config", 
            "osc-pose-controller.yml"
        )
        
        # Motion parameters
        self.action_scale = config.get("action_scale", {
            "translation": 0.05,
            "rotation": 0.05
        })
        
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
        
        # Robot interface
        self.robot_interface = None
        self.controller_cfg = None
        self.ik_solver = None
        self._current_state = None
        self._is_connected = False
        self._error_state = False
        self._gripper_state = np.array([0.0])  # 0: open, 1: closed
        
        # Default initial joint positions
        self.init_qpos = config.get("init_qpos", [
            0.09162008,
            -0.19826458,
            -0.01990020,
            -2.47322699,
            -0.01307074,
            2.30396583,
            0.84809397
        ])
        
    def connect(self) -> bool:
        """Connect to the Franka Panda robot using deoxys."""
        if not DEOXYS_AVAILABLE:
            print("deoxys is not installed. Cannot connect to Franka Panda.")
            return False
            
        try:
            # Get config path
            if os.path.exists(self.deoxys_config_path):
                config_path = self.deoxys_config_path
            else:
                # Try relative to deoxys config root
                config_path = os.path.join(config_root, self.deoxys_config_path)
                if not os.path.exists(config_path):
                    config_path = self.deoxys_config_path  # Use as-is and let deoxys handle
            
            # Initialize Franka interface from deoxys
            self.robot_interface = FrankaInterface(
                general_cfg_file=config_path,
                control_freq=self.control_freq,
                state_freq=self.state_freq,
                has_gripper=True,
                use_visualizer=self.use_visualizer
            )
            
            # Load controller configuration
            if os.path.exists(self.controller_config_file):
                controller_cfg_path = self.controller_config_file
            else:
                controller_cfg_path = os.path.join(config_root, self.controller_config_file)
            
            if os.path.exists(controller_cfg_path):
                self.controller_cfg = YamlConfig(controller_cfg_path).as_easydict()
            else:
                # Use default controller config
                self.controller_cfg = get_default_controller_config(self.default_controller_type)
            
            # Override action scales if specified
            if hasattr(self.controller_cfg, 'action_scale'):
                if 'translation' in self.action_scale:
                    self.controller_cfg.action_scale.translation = self.action_scale['translation']
                if 'rotation' in self.action_scale:
                    self.controller_cfg.action_scale.rotation = self.action_scale['rotation']
            
            # Initialize IK solver
            self.ik_solver = IKSolver(self.config)
            
            # Wait for robot state to be available
            time.sleep(0.5)
            
            self._is_connected = True
            self._error_state = False
            print(f"Successfully connected to Franka Panda robot via deoxys")
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
            if self.robot_interface is not None:
                # Send termination signal
                self.robot_interface.control(
                    controller_type=self.default_controller_type,
                    action=np.zeros(7),
                    controller_cfg=self.controller_cfg,
                    termination=True
                )
                
                # Close interface
                self.robot_interface.close()
                self.robot_interface = None
            
            self.ik_solver = None
            self._is_connected = False
            print("Disconnected from Franka Panda robot")
            return True
            
        except Exception as e:
            print(f"Error during Franka Panda disconnect: {e}")
            return False
    
    def reset_to_init(self) -> bool:
        """Reset the robot to the initial configuration."""
        if self.robot_interface is not None:
            try:
                print("Resetting to initial position...")
                
                # Use joint position controller for reset
                controller_type = "JOINT_POSITION"
                
                # Get appropriate controller config
                joint_controller_cfg = get_default_controller_config(controller_type)
                
                # Add gripper action (open)
                action = self.init_qpos + [-1.0]  # -1.0 for open gripper
                
                # Move to initial position
                max_iterations = 200
                for _ in range(max_iterations):
                    # Check if we've reached the target
                    if len(self.robot_interface._state_buffer) > 0:
                        current_q = np.array(self.robot_interface._state_buffer[-1].q)
                        if np.max(np.abs(current_q - np.array(self.init_qpos))) < 1e-3:
                            break
                    
                    # Send control command
                    self.robot_interface.control(
                        controller_type=controller_type,
                        action=action,
                        controller_cfg=joint_controller_cfg,
                    )
                
                self._gripper_state = np.array([0.0])  # Gripper open
                print("Reset finished!")
                return True
                
            except Exception as e:
                print(f"Error during reset: {e}")
                return False
        return False
    
    def move_to_joint_positions(self, joint_positions: np.ndarray) -> bool:
        """Move robot to specified joint positions."""
        if self.robot_interface is not None:
            try:
                print(f"Moving to joint positions {joint_positions}")
                
                controller_type = "JOINT_POSITION"
                joint_controller_cfg = get_default_controller_config(controller_type)
                
                # Add gripper action (maintain current state)
                gripper_action = 1.0 if self._gripper_state.item() else -1.0
                action = joint_positions.tolist() + [gripper_action]
                
                # Move to target position
                max_iterations = 200
                for _ in range(max_iterations):
                    if len(self.robot_interface._state_buffer) > 0:
                        current_q = np.array(self.robot_interface._state_buffer[-1].q)
                        if np.max(np.abs(current_q - joint_positions)) < 1e-3:
                            break
                    
                    self.robot_interface.control(
                        controller_type=controller_type,
                        action=action,
                        controller_cfg=joint_controller_cfg,
                    )
                
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
            # Get latest state from deoxys interface
            if len(self.robot_interface._state_buffer) == 0:
                # Return last known state if buffer is empty
                if self._current_state is not None:
                    return self._current_state
                else:
                    raise RuntimeError("No robot state available")
            
            # Get the latest state
            deoxys_state = self.robot_interface._state_buffer[-1]
            
            # Extract joint information
            joint_positions = np.array(deoxys_state.q)
            joint_velocities = np.array(deoxys_state.dq)
            joint_torques = np.array(deoxys_state.tau_J)
            
            # Get end-effector pose from deoxys state
            # deoxys provides O_T_EE as a 16-element list (4x4 matrix)
            ee_pose = np.array(deoxys_state.O_T_EE).reshape(4, 4)
            tcp_position = ee_pose[:3, 3]
            
            # Convert rotation matrix to quaternion
            from scipy.spatial.transform import Rotation
            rotation = Rotation.from_matrix(ee_pose[:3, :3])
            tcp_orientation = rotation.as_quat()  # [x, y, z, w]
            
            # Get gripper state
            gripper_state = self._gripper_state
            
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
            # Determine controller type based on action
            if action.control_mode == "joint":
                if action.joint_positions is not None:
                    controller_type = "JOINT_POSITION"
                    controller_cfg = get_default_controller_config(controller_type)
                    
                    # Add gripper action
                    gripper_action = 1.0 if self._gripper_state.item() else -1.0
                    deoxys_action = action.joint_positions.tolist() + [gripper_action]
                    
                elif action.joint_velocities is not None:
                    # Deoxys doesn't have direct joint velocity control
                    # Convert to position control
                    current_q = np.array(self.robot_interface.last_q)
                    dt = 1.0 / self.control_freq
                    target_q = current_q + action.joint_velocities * dt
                    
                    # Clip to joint limits
                    target_q = np.clip(target_q, self.joint_limits["min"], self.joint_limits["max"])
                    
                    controller_type = "JOINT_POSITION"
                    controller_cfg = get_default_controller_config(controller_type)
                    gripper_action = 1.0 if self._gripper_state.item() else -1.0
                    deoxys_action = target_q.tolist() + [gripper_action]
                    
                elif action.joint_torques is not None:
                    # Use joint impedance controller for torque-like control
                    controller_type = "JOINT_IMPEDANCE"
                    controller_cfg = get_default_controller_config(controller_type)
                    
                    # Scale torques to impedance actions
                    gripper_action = 1.0 if self._gripper_state.item() else -1.0
                    deoxys_action = (action.joint_torques * 0.01).tolist() + [gripper_action]
                    
            elif action.control_mode == "cartesian":
                if (action.tcp_position is not None and 
                    action.tcp_orientation is not None):
                    
                    # Use OSC_POSE controller
                    controller_type = "OSC_POSE"
                    controller_cfg = self.controller_cfg.copy() if self.controller_cfg else get_default_controller_config(controller_type)
                    
                    # Get current pose
                    current_state = self.get_state()
                    
                    # Calculate deltas
                    pos_delta = action.tcp_position - current_state.tcp_position
                    
                    # Convert quaternions to axis-angle for rotation delta
                    from scipy.spatial.transform import Rotation
                    current_rot = Rotation.from_quat(current_state.tcp_orientation)
                    target_rot = Rotation.from_quat(action.tcp_orientation)
                    rot_delta = target_rot * current_rot.inv()
                    axis_angle = rot_delta.as_rotvec()
                    
                    # Create action [dx, dy, dz, dax, day, daz, gripper]
                    gripper_action = 1.0 if self._gripper_state.item() else -1.0
                    deoxys_action = np.concatenate([
                        pos_delta,
                        axis_angle,
                        [gripper_action]
                    ])
                    
                    # Set as delta action
                    controller_cfg.is_delta = True
            
            # Send control command
            self.robot_interface.control(
                controller_type=controller_type,
                action=deoxys_action,
                controller_cfg=controller_cfg,
            )
            
            # Handle gripper state update
            if action.gripper_state is not None:
                if action.gripper_state.item() != self._gripper_state.item():
                    self._gripper_state = action.gripper_state
                    print(f"Gripper state updated to: {'closed' if action.gripper_state.item() else 'open'}")
            
            return True
            
        except Exception as e:
            print(f"Error sending action: {e}")
            traceback.print_exc()
            self._error_state = True
            return False
    
    def is_connected(self) -> bool:
        """Check if the robot is connected."""
        return self._is_connected and not self._error_state and self.robot_interface is not None
    
    def is_error(self) -> bool:
        """Check if the robot is in an error state."""
        if not self._is_connected:
            return True
        return self._error_state
    
    def clear_error(self) -> bool:
        """Clear any error state."""
        if not self._is_connected:
            return False
            
        try:
            # Reset error state
            self._error_state = False
            
            # Reinitialize controller if needed
            if self.robot_interface is not None:
                self.robot_interface.preprocess()
            
            return True
        except Exception as e:
            print(f"Error clearing robot error: {e}")
            return False
    
    def stop(self) -> bool:
        """Emergency stop the robot."""
        if not self._is_connected:
            return False
            
        try:
            if self.robot_interface is not None:
                # Send termination signal
                self.robot_interface.control(
                    controller_type=self.default_controller_type,
                    action=np.zeros(7),
                    controller_cfg=self.controller_cfg,
                    termination=True
                )
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
        try:
            # Use IK solver if available
            if self.ik_solver is not None:
                return self.ik_solver.get_tcp_pose(joint_positions)
            else:
                # Fallback: compute from deoxys kinematics if available
                # This would require the robot model which deoxys has internally
                raise NotImplementedError("Forward kinematics requires IK solver")
                    
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
        try:
            if self.ik_solver is not None:
                joint_solution, success = self.ik_solver.solve(
                    target_position=target_position,
                    target_orientation=target_orientation,
                    dt=1.0 / self.control_freq,
                    current_q=current_q,
                )
                return joint_solution if success else None
            else:
                print("No IK solver available")
                return None
                    
        except Exception as e:
            print(f"Error in inverse kinematics: {e}")
            self._error_state = True
            return None