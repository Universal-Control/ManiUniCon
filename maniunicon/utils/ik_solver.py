#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Unified inverse kinematics solver module for robot control."""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pinocchio as pin
import pink
import qpsolvers
from pink import Configuration, solve_ik
from pink.limits import ConfigurationLimit, VelocityLimit
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from robot_descriptions.loaders.pinocchio import load_robot_description


class IKSolver:
    """Unified inverse kinematics solver using Pink library.

    This class provides a common interface for solving inverse kinematics
    for both simulation and real robot control.
    """

    def __init__(
        self,
        robot_config: Dict[str, Any],
        robot_model: Optional[pin.Model] = None,
        robot_data: Optional[pin.Data] = None,
    ):
        """Initialize the IK solver.

        Args:
            robot_config: Robot configuration dictionary containing:
                - urdf_path: Path to URDF file (optional)
                - robot_name: Robot name for loading from robot_descriptions (optional)
                - tcp_frame: End-effector frame name
                - ik_damping: Damping factor for IK solver
                - ik_solver: QP solver name (e.g., "daqp", "proxqp", "osqp")
                - init_configuration: Initial joint configuration
                - enable_posture_task: Whether to include posture task
                - posture_cost: Cost for posture task
                - position_cost: Cost for position tracking
                - orientation_cost: Cost for orientation tracking
            robot_model: Pre-loaded Pinocchio model (optional)
            robot_data: Pre-loaded Pinocchio data (optional)
        """
        self.robot_config = robot_config

        # Load robot model if not provided
        if robot_model is None or robot_data is None:
            self._load_robot_model()
        else:
            self.model = robot_model
            self.data = robot_data

        # Initialize configuration
        self._init_configuration()

        # Create tasks
        self._create_tasks()

        # Create limits
        self._create_limits()

        # IK solver parameters
        self.ik_damping = robot_config.get("ik_damping", 1e-12)
        self.ik_solver = self._select_qp_solver(robot_config.get("ik_solver", None))

    def _load_robot_model(self):
        """Load robot model from URDF or robot_descriptions."""
        if self.robot_config.get("urdf_path") is not None:
            robot = pin.RobotWrapper.BuildFromURDF(
                filename=self.robot_config.get("urdf_path"),
                package_dirs=list(self.robot_config.get("urdf_package_dirs", None)),
                root_joint=None,
            )
        else:
            robot = load_robot_description(
                self.robot_config["robot_name"], root_joint=None
            )
        self.model = robot.model
        self.data = robot.data

    def _init_configuration(self):
        """Initialize robot configuration."""
        # Get initial configuration
        init_config = self.robot_config.get("init_configuration", {})

        # Create configuration vector
        if hasattr(pink.utils, "custom_configuration_vector"):
            q_ref = custom_configuration_vector(
                pin.RobotWrapper(self.model, collision_model=None, visual_model=None),
                **init_config,
            )
        else:
            # Fallback to default configuration
            q_ref = pin.neutral(self.model)

        self.configuration = Configuration(self.model, self.data, q_ref)

    def _create_tasks(self):
        """Create IK tasks."""
        # Create end-effector task
        self.ee_task = FrameTask(
            self.robot_config.get("tcp_frame", "wrist_3_link"),
            position_cost=self.robot_config.get("position_cost", 1.0),
            orientation_cost=self.robot_config.get("orientation_cost", 1.0),
        )

        # Create task list
        self.tasks = [self.ee_task]

        # Add posture task if enabled
        if self.robot_config.get("enable_posture_task", True):
            self.posture_task = PostureTask(
                cost=self.robot_config.get("posture_cost", 1e-3),
            )
            self.tasks.append(self.posture_task)

        # Set initial targets from configuration
        for task in self.tasks:
            task.set_target_from_configuration(self.configuration)

    def _create_limits(self):
        """Create joint limits."""
        self.limits = []

        # Configuration limits
        if self.robot_config.get("enable_configuration_limits", True):
            self.configuration_limit = ConfigurationLimit(
                self.model,
                config_limit_gain=self.robot_config.get("config_limit_gain", 0.5),
            )
            self.limits.append(self.configuration_limit)

        # Velocity limits
        if self.robot_config.get("enable_velocity_limits", True):
            self.velocity_limit = VelocityLimit(
                self.model,
            )
            self.limits.append(self.velocity_limit)

    def _select_qp_solver(self, preferred_solver: Optional[str] = None) -> str:
        """Select QP solver based on availability and preference."""
        available = qpsolvers.available_solvers

        if not available:
            raise RuntimeError("No QP solvers available")

        # Use preferred solver if available
        if preferred_solver and preferred_solver in available:
            return preferred_solver

        # Prefer daqp if available
        if "daqp" in available:
            return "daqp"

        # Fallback to first available solver
        return available[0]

    def set_current_configuration(self, q: np.ndarray):
        """Update current joint configuration.

        Args:
            q: Joint configuration vector
        """
        self.configuration.update(q)
        pin.forwardKinematics(self.model, self.data, self.configuration.q)
        pin.updateFramePlacements(self.model, self.data)

    def solve(
        self,
        target_position: Optional[np.ndarray] = None,
        target_orientation: Optional[np.ndarray] = None,
        dt: float = 0.01,
        current_q: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """Solve inverse kinematics.

        Args:
            target_position: Target position (3D vector)
            target_orientation: Target orientation as quaternion [x, y, z, w]
            dt: Time step for integration
            current_q: Current joint configuration (if different from internal state)

        Returns:
            Tuple of (joint configuration, success flag)
        """
        # Update current configuration if provided
        if current_q is not None:
            self.set_current_configuration(current_q)

        # Update target pose
        if target_position is not None and target_orientation is not None:
            # Create SE3 target
            target_se3 = pin.SE3(
                pin.Quaternion(target_orientation),
                target_position,
            )
            self.ee_task.set_target(target_se3)

        try:
            # Solve IK
            velocity = solve_ik(
                self.configuration,
                self.tasks,
                dt,
                self.ik_solver,
                damping=self.ik_damping,
                limits=self.limits if self.limits else None,
            )

            # Integrate velocity
            self.configuration.integrate_inplace(velocity, dt)

            return self.configuration.q.copy(), True

        except Exception as e:
            print(f"IK solver error: {e}")
            return self.configuration.q.copy(), False

    def get_tcp_pose(
        self, q: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get TCP pose for given or current configuration.

        Args:
            q: Joint configuration (uses current if None)

        Returns:
            Tuple of (position, orientation as quaternion [x, y, z, w])
        """
        if q is not None:
            # Use a separate data object to avoid interference with ongoing IK computations
            temp_data = pin.Data(self.model)
            pin.forwardKinematics(self.model, temp_data, q)
            pin.updateFramePlacements(self.model, temp_data)

            frame_id = self.model.getFrameId(self.robot_config["tcp_frame"])
            frame_pose = temp_data.oMf[frame_id]
        else:
            frame_id = self.model.getFrameId(self.robot_config["tcp_frame"])
            frame_pose = self.data.oMf[frame_id]

        position = frame_pose.translation
        orientation = pin.Quaternion(frame_pose.rotation).coeffs()  # [x, y, z, w]

        return position.copy(), orientation.copy()

    def get_jacobian(self, q: Optional[np.ndarray] = None) -> np.ndarray:
        """Get end-effector Jacobian matrix.

        Args:
            q: Joint configuration (uses current if None)

        Returns:
            6xN Jacobian matrix
        """
        if q is not None:
            self.set_current_configuration(q)

        frame_id = self.model.getFrameId(self.robot_config["tcp_frame"])
        return pin.computeFrameJacobian(
            self.model,
            self.data,
            self.configuration.q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
