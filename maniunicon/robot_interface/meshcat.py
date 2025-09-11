#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Meshcat-based simulation interface for robot debugging."""

import importlib
import time
import traceback
import threading
from typing import Any, Dict

import numpy as np
import pinocchio as pin
from pink.visualization import start_meshcat_visualizer
from robot_descriptions.loaders.pinocchio import load_robot_description
from robot_descriptions._package_dirs import get_package_dirs

from maniunicon.robot_interface.base import RobotInterface
from maniunicon.utils import meshcat_shapes
from maniunicon.utils.ik_solver import IKSolver
from maniunicon.utils.shared_memory.shared_storage import RobotAction, RobotState
from maniunicon.utils.ruckig_utils import init_ruckig, update_ruckig


class MeshcatInterface(RobotInterface):
    """Meshcat-based simulation interface for robot debugging.

    This interface provides a visual simulation of the robot using meshcat,
    which is useful for debugging and testing before deploying on real hardware.
    It implements the same interface as the real robot, but uses a physics
    simulation instead of actual hardware.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the meshcat simulation interface.

        Args:
            config: Dictionary containing robot configuration parameters
        """
        super().__init__(config)

        # Simulation components
        self.robot = None
        self.model = None
        self.data = None
        self.fk_data = None  # Separate data object for forward kinematics
        self.viz = None
        self.viewer = None
        self.ik_solver = None
        self.configuration = None

        # Simulation state
        self._current_state = None
        self._is_connected = False
        self._error_state = False
        self.gripper_state = np.array([0.0])
        self.dt = 1.0 / config.get("frequency", 200.0)

        # Thread safety for visualization - will be initialized in connect()
        self._viz_lock = None

    def _init_threading(self):
        """Initialize threading components. Called after process starts to avoid pickling issues."""
        if self._viz_lock is None:
            self._viz_lock = threading.Lock()

    def connect(self) -> bool:
        """Connect to the simulated robot."""
        try:
            # Initialize threading components
            self._init_threading()

            # Load robot model
            if self.config.get("urdf_path") is not None:
                self.robot = pin.RobotWrapper.BuildFromURDF(
                    filename=self.config.get("urdf_path", None),
                    package_dirs=list(self.config.get("urdf_package_dirs", None)),
                    root_joint=None,
                )
            else:
                if self.config["robot_name"] == "ur5_description_gripper":
                    module = importlib.import_module(
                        "robot_descriptions.ur5_description"
                    )
                    self.robot = pin.RobotWrapper.BuildFromURDF(
                        filename=module.URDF_PATH_GRIPPER,
                        package_dirs=get_package_dirs(module),
                        root_joint="base_link",
                    )
                else:
                    self.robot = load_robot_description(
                        self.config["robot_name"], root_joint=None
                    )

            self.model = self.robot.model
            self.data = self.robot.data
            self.fk_data = pin.Data(self.model)  # Initialize FK data object

            self.viz = start_meshcat_visualizer(self.robot, open=False)
            self.viewer = self.viz.viewer
            # Add visualization frames for end-effector and target
            meshcat_shapes.frame(self.viewer["end_effector_target"], opacity=0.5)
            meshcat_shapes.frame(self.viewer["end_effector"], opacity=1.0)
            print("Meshcat visualization initialized successfully")

            # Initialize IK solver
            self.ik_solver = IKSolver(self.config, self.model, self.data)

            # Get initial configuration
            self.configuration = self.ik_solver.configuration

            self._is_connected = True
            self._error_state = False

            if self.config.get("use_ruckig", False):
                # Initialize Ruckig
                self.otg, self.otg_inp, self.otg_out, self.otg_res = init_ruckig(
                    self.configuration.q, np.zeros(self.config["num_joints"]), self.dt
                )
                self.last_command_time = time.time()

            print("Successfully connected to meshcat simulation")
            return True

        except Exception as e:
            print(f"Error connecting to meshcat simulation: {e}")
            print(traceback.format_exc())
            self._is_connected = False
            self._error_state = True
            return False

    def disconnect(self) -> bool:
        """Disconnect from the simulated robot."""
        try:
            # Clean up visualization with thread safety
            if self._viz_lock is not None:
                with self._viz_lock:
                    if self.viz is not None:
                        try:
                            # Try to close meshcat connection gracefully
                            if hasattr(self.viz, "close"):
                                self.viz.close()
                        except Exception as e:
                            print(f"Warning: Error closing meshcat visualization: {e}")
                        finally:
                            self.viz = None
                            self.viewer = None
            else:
                # If no lock available, just clear references
                self.viz = None
                self.viewer = None

            # Clear other references
            self.robot = None
            self.model = None
            self.data = None
            self.fk_data = None  # Clear FK data
            self.ik_solver = None
            self.configuration = None

            self._is_connected = False
            print("Disconnected from meshcat simulation")
            return True

        except Exception as e:
            print(f"Error during meshcat disconnect: {e}")
            return False

    def get_state(self) -> RobotState:
        """Get the current state of the simulated robot."""
        if not self.is_connected():
            raise RuntimeError("Robot is not connected")

        try:
            # Get TCP pose using IK solver
            tcp_position, tcp_orientation = self.ik_solver.get_tcp_pose(
                self.configuration.q
            )

            current_state = RobotState(
                joint_positions=self.configuration.q,
                joint_velocities=np.zeros(self.config["num_joints"]),
                joint_torques=np.zeros(self.config["num_joints"]),
                tcp_position=tcp_position,
                tcp_orientation=tcp_orientation,
                gripper_state=self.gripper_state,
                timestamp=np.array(time.time()),
            )

            self._current_state = current_state

            # Update visualization safely
            self._update_visualization()

            return current_state

        except Exception as e:
            print(f"Error getting robot state: {e}")
            import traceback

            traceback.print_exc()
            self._error_state = True
            if self._current_state is None:
                raise RuntimeError("No robot state available")
            return self._current_state

    def send_action(self, action: RobotAction) -> bool:
        """Send a control action to the simulated robot."""
        if not hasattr(self, "last_action_timestamp"):
            self.last_action_timestamp = time.time()
        else:
            print(
                f"Time since last action: {(time.time() - self.last_action_timestamp) * 1000} ms"
            )
            self.last_action_timestamp = time.time()

        if not self.is_connected():
            return False

        if not self.validate_action(action):
            print("Invalid action: exceeds robot limits")
            return False

        try:
            if action.control_mode == "joint":
                # Direct joint control
                if action.joint_positions is not None:
                    self._update_joint_positions(np.array(action.joint_positions))
                    self.ik_solver.set_current_configuration(self.configuration.q)
                elif action.joint_velocities is not None:
                    self.configuration.integrate_inplace(
                        np.array(action.joint_velocities), self.dt
                    )
                    self.ik_solver.set_current_configuration(self.configuration.q)

            elif action.control_mode == "cartesian":
                # Cartesian control
                if (
                    action.tcp_position is not None
                    and action.tcp_orientation is not None
                ):
                    # Get current TCP pose for partial updates
                    target_position = action.tcp_position
                    target_orientation = action.tcp_orientation

                    # Solve IK
                    q_solution, success = self.ik_solver.solve(
                        target_position=np.array(target_position),
                        target_orientation=np.array(target_orientation),
                        dt=self.dt,
                    )

                    if success:
                        self._update_joint_positions(q_solution)
                    else:
                        print("IK solution failed")
                        return False

            # Update gripper state
            self.gripper_state = action.gripper_state
            # TODO(zbzhu): change gripper state in meshcat visualization

            # Update visualization safely
            self._update_visualization()

            return True

        except Exception as e:
            print(f"Error sending action: {e}")
            import traceback

            traceback.print_exc()
            self._error_state = True
            return False

    def _update_visualization(self):
        """Update the meshcat visualization with thread safety."""
        # Use lock to prevent concurrent visualization updates if available
        if self._viz_lock is not None:
            with self._viz_lock:
                self._do_visualization_update()
        else:
            self._do_visualization_update()

    def _update_joint_positions(self, q):
        if self.config.get("use_ruckig", False):
            q, dq_d, last_command_time = update_ruckig(
                self.otg,
                self.otg_inp,
                self.otg_out,
                self.otg_res,
                q,
                self.last_command_time,
                self.dt,
            )
            self.last_command_time = last_command_time
            q = np.array(q)
        if q is None:
            q = self.configuration.q
        self.configuration.q = q

    def _do_visualization_update(self):
        """Perform the actual visualization update."""
        if self.viewer is not None and self.ik_solver is not None:
            # Check if meshcat connection is still alive
            self.viewer["end_effector_target"].set_transform(
                self.ik_solver.ee_task.transform_target_to_world.np
            )
            self.viewer["end_effector"].set_transform(
                self.configuration.get_transform_frame_to_world(
                    self.ik_solver.ee_task.frame
                ).np
            )
            self.viz.display(self.configuration.q)

    def reset_to_init(self):
        """Reset the robot to the initial configuration."""
        if self.configuration is not None and "init_qpos" in self.config:
            init_qpos = self.config.get("init_qpos", None)
            self.configuration.q = np.array(init_qpos)
            self.ik_solver.set_current_configuration(self.configuration.q)
            self._update_visualization()
            return True
        return False

    def is_connected(self) -> bool:
        """Check if the robot is connected."""
        return self._is_connected and not self._error_state

    def is_error(self) -> bool:
        """Check if the robot is in an error state."""
        if not self._is_connected:
            return True

        return self._error_state

    def clear_error(self) -> bool:
        """Clear any error state."""
        if not self._is_connected:
            return False

        self._error_state = False
        return True

    def stop(self) -> bool:
        """Emergency stop the robot."""
        if not self._is_connected:
            return False

        try:
            # Send zero velocity command
            action = RobotAction(
                control_mode="joint",
                joint_velocities=np.zeros(self.num_joints),
                timestamp=time.time(),
            )
            return self.send_action(action)
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
            traceback.print_exc()
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
            traceback.print_exc()
            self._error_state = True
            return None
