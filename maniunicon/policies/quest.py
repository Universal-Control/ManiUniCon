#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Quest-based teleoperation policy for robot control."""

import os
import time
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation as R
from multiprocessing.synchronize import Event
import traceback
from loop_rate_limiters import RateLimiter
from maniunicon.utils.data import get_next_episode_dir
from pynput import keyboard

from maniunicon.utils.shared_memory.shared_storage import (
    SharedStorage,
    RobotAction,
    RobotState,
)
from maniunicon.core.policy import BasePolicy
from maniunicon.utils.quest_controller import VRPolicy


class QuestPolicy(BasePolicy):
    """Quest-based teleoperation policy.

    Controls:
    - Translation: Move end-effector in 3D space using Quest controller position
    - Rotation: Rotate end-effector using Quest controller orientation
    - Right Grip Button (RG): Enable/disable movement control
    - Right Trigger (RTr): Toggle gripper open/close
    - A Button: Toggle recording (start/stop episode recording)
    - B Button: Drop current episode without saving
    - 'r' key: Reset robot to initial state

    Safety Features:
    - Delta limiting: Prevents large position/rotation changes per step
    - Emergency stop: Stops robot movement when safety is violated
    """

    def __init__(
        self,
        shared_storage: SharedStorage,
        reset_event: Event,
        pos_scaling_factor: float = 1.0,
        rot_scaling_factor: float = 0.5,
        ee_trans_mat: np.ndarray | list = np.eye(3),
        control_interval: int = 1,  # Number of action steps to send
        dt: float = 0.01,  # Time step between actions
        command_latency: float = 0.005,  # seconds
        record_dir: Optional[str] = None,
        workspace_bounds: dict = None,
        name: str = "QuestPolicy",
        # Safety parameters
        max_delta_pos: float = 0.1,  # Maximum allowed position change per step (meters)
        max_delta_rot: float = 0.5,  # Maximum allowed rotation change per step (radians)
    ):
        super().__init__(
            shared_storage=shared_storage,
            reset_event=reset_event,
            command_latency=command_latency,
            name=name,
        )
        self.pos_scaling_factor = pos_scaling_factor
        self.rot_scaling_factor = rot_scaling_factor
        self.ee_trans_mat = (
            np.array(ee_trans_mat) if isinstance(ee_trans_mat, list) else ee_trans_mat
        )
        self.control_interval = control_interval
        self.dt = dt
        self.frequency = 1 / dt
        self.record_dir = record_dir
        self.workspace_bounds = workspace_bounds

        # Safety parameters
        self.max_delta_pos = max_delta_pos
        self.max_delta_rot = max_delta_rot

        # Internal state
        self._vr_policy: Optional[VRPolicy] = None
        self._recording_active = False
        self._current_episode_dir: Optional[str] = None
        self._a_button_pressed = (
            False  # Track A button state to avoid multiple triggers
        )
        self._b_button_pressed = (
            False  # Track B button state to avoid multiple triggers
        )
        self._listener: Optional[keyboard.Listener] = None

        # Cached robot state
        self._current_tcp_position = None
        self._current_tcp_orientation = None
        self._current_joint_positions = None

    def sync_state(self):
        """Sync the state of the robot with the shared storage."""
        state = None
        while state is None:
            state: RobotState | None = self.shared_storage.read_state(k=1)
            print("QuestPolicy: waiting for state...")
            time.sleep(0.05)

        self._current_tcp_position = state.tcp_position[0].copy()
        self._current_tcp_orientation = state.tcp_orientation[0].copy()
        self._current_joint_positions = state.joint_positions[0].copy()

    def _clip_tcp_pose_to_bounds(
        self, tcp_position: np.ndarray, tcp_orientation: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Clip TCP pose to stay within workspace bounds."""
        if self.workspace_bounds is None or not self.workspace_bounds.get(
            "enabled", False
        ):
            return tcp_position, tcp_orientation

        # Clip position bounds
        clipped_position = tcp_position.copy()
        pos_bounds = self.workspace_bounds.get("position", {})
        if pos_bounds.get("x_min") is not None:
            clipped_position[0] = max(clipped_position[0], pos_bounds["x_min"])
        if pos_bounds.get("x_max") is not None:
            clipped_position[0] = min(clipped_position[0], pos_bounds["x_max"])
        if pos_bounds.get("y_min") is not None:
            clipped_position[1] = max(clipped_position[1], pos_bounds["y_min"])
        if pos_bounds.get("y_max") is not None:
            clipped_position[1] = min(clipped_position[1], pos_bounds["y_max"])
        if pos_bounds.get("z_min") is not None:
            clipped_position[2] = max(clipped_position[2], pos_bounds["z_min"])
        if pos_bounds.get("z_max") is not None:
            clipped_position[2] = min(clipped_position[2], pos_bounds["z_max"])

        # Clip orientation bounds
        clipped_orientation = tcp_orientation.copy()
        ori_bounds = self.workspace_bounds.get("orientation", {})
        if any(
            ori_bounds.get(key) is not None
            for key in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        ):
            # Convert quaternion to euler angles for bounds checking
            rotation = R.from_quat(tcp_orientation)
            roll, pitch, yaw = rotation.as_euler("xyz", degrees=False)

            # Clip roll (X rotation)
            if ori_bounds.get("x_min") is not None:
                roll = max(roll, ori_bounds["x_min"])
            if ori_bounds.get("x_max") is not None:
                roll = min(roll, ori_bounds["x_max"])

            # Clip pitch (Y rotation)
            if ori_bounds.get("y_min") is not None:
                pitch = max(pitch, ori_bounds["y_min"])
            if ori_bounds.get("y_max") is not None:
                pitch = min(pitch, ori_bounds["y_max"])

            # Clip yaw (Z rotation)
            if ori_bounds.get("z_min") is not None:
                yaw = max(yaw, ori_bounds["z_min"])
            if ori_bounds.get("z_max") is not None:
                yaw = min(yaw, ori_bounds["z_max"])

            # Convert back to quaternion
            clipped_rotation = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
            clipped_orientation = clipped_rotation.as_quat()

        return clipped_position, clipped_orientation

    def _on_press(self, key):
        """Handle key press events."""
        try:
            # Handle robot reset on 'r' key press
            if key == keyboard.KeyCode.from_char("r"):
                print("Robot reset triggered by 'r' key")
                # Trigger reset by setting the reset event
                if self.reset_event is not None:
                    self.reset_event.set()
        except AttributeError:
            pass

    def _on_release(self, key):
        """Handle key release events."""
        try:
            # No specific release handling needed for reset
            pass
        except AttributeError:
            pass

    def run(self):
        """Main process loop."""
        try:
            # Connect to Quest controller
            try:
                self._vr_policy = VRPolicy(
                    pos_scaling_factor=self.pos_scaling_factor,
                    rot_scaling_factor=self.rot_scaling_factor,
                    ee_trans_mat=self.ee_trans_mat,
                    # Safety parameters
                    max_delta_pos=self.max_delta_pos,
                    max_delta_rot=self.max_delta_rot,
                )
                self._vr_policy.start()
                print("Quest controller connected successfully")
                print("\nQuest controller controls:")
                print(
                    "Translation: Move end-effector in 3D space using controller position"
                )
                print("Rotation: Rotate end-effector using controller orientation")
                print("Right Grip Button (RG): Enable/disable movement control")
                print("Right Trigger (RTr): Toggle gripper open/close")
                print("A Button: Toggle recording (start/stop episode recording)")
                print("B Button: Drop current episode without saving")
                print("'r' key: Reset robot to initial state")
                print("\nSafety features enabled:")
                print(f"  - Max position delta: {self.max_delta_pos:.3f}m per step")
                print(f"  - Max rotation delta: {self.max_delta_rot:.3f}rad per step")
                if self.record_dir is not None:
                    print(f"\nRecording directory: {self.record_dir}")
                self.sync_state()
            except Exception as e:
                print(f"Failed to connect Quest controller: {e}")
                traceback.print_exc()
                self.shared_storage.error_state.value = True
                return

            # Start keyboard listener
            self._listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self._listener.start()

            rate = RateLimiter(frequency=self.frequency, warn=True, name="quest_policy")

            while self.shared_storage.is_running.value:
                if self.reset_event is not None and self.reset_event.is_set():
                    # Clear actions during reset
                    self.shared_storage.read_all_action()
                    self.sync_state()
                    # Reset VR policy state
                    self._vr_policy.reset_state()
                    rate.sleep()
                    continue

                # Prepare poses for VR policy
                poses = {
                    "translation": self._current_tcp_position,
                    "rotation": R.from_quat(self._current_tcp_orientation).as_matrix(),
                }

                # Get action from VR policy
                action_dict = self._vr_policy.forward(poses)

                # Handle reset command
                if action_dict is None:
                    # No VR data available, maintain current pose
                    print("No VR action available")
                    action_dict = {
                        "position": self._current_tcp_position,
                        "rmat": R.from_quat(self._current_tcp_orientation).as_matrix(),
                        "gripper": np.array([0.0]),  # Default gripper state
                        "a_button_pressed": False,
                        "b_button_pressed": False,
                    }

                # Handle A button press for recording toggle
                a_button_pressed = action_dict.get("a_button_pressed", False)
                if a_button_pressed and not self._a_button_pressed:
                    # A button was just pressed
                    self._recording_active = not self._recording_active
                    print(f"Recording: {'ON' if self._recording_active else 'OFF'}")

                self._a_button_pressed = a_button_pressed

                # Handle B button press for dropping current episode
                b_button_pressed = action_dict.get("b_button_pressed", False)
                if (
                    b_button_pressed
                    and not self._b_button_pressed
                    and self._recording_active
                ):
                    # B button was just pressed and recording is active
                    self._recording_active = False
                    # First stop recording in shared storage
                    self.shared_storage.clear_record_dir()
                    # Then remove the directory if it exists
                    if self._current_episode_dir and os.path.exists(
                        self._current_episode_dir
                    ):
                        import shutil

                        shutil.rmtree(self._current_episode_dir)
                        print(
                            f"Dropped episode - Removed directory: {self._current_episode_dir}"
                        )
                    self._current_episode_dir = None
                    self.shared_storage.stop_record()
                    print("Recording: OFF")

                self._b_button_pressed = b_button_pressed

                # Handle recording state changes
                if (
                    self._recording_active
                    and not self.shared_storage.is_recording.value
                ):
                    # Start recording
                    if self.record_dir is not None:
                        self._current_episode_dir = get_next_episode_dir(
                            self.record_dir
                        )
                        self.shared_storage.set_record_dir(self._current_episode_dir)
                        self.shared_storage.start_record(
                            start_time=time.time(),
                            dt=self.dt,
                        )
                        print(
                            f"Recording started - Episode: {os.path.basename(self._current_episode_dir)}"
                        )
                    else:
                        print(
                            "Recording directory not specified. Cannot start recording."
                        )
                        self._recording_active = False
                elif (
                    not self._recording_active
                    and self.shared_storage.is_recording.value
                ):
                    # Stop recording
                    self.shared_storage.stop_record()
                    if self._current_episode_dir:
                        print(
                            f"Recording stopped - Episode saved to: {self._current_episode_dir}"
                        )
                    else:
                        print("Recording stopped")
                    self._current_episode_dir = None

                # Update current pose based on VR action
                # Update position and orientation
                self._current_tcp_position = action_dict["position"]
                self._current_tcp_orientation = R.from_matrix(
                    action_dict["rmat"]
                ).as_quat()

                # Apply workspace bounds clipping to internal state
                self._current_tcp_position, self._current_tcp_orientation = (
                    self._clip_tcp_pose_to_bounds(
                        self._current_tcp_position, self._current_tcp_orientation
                    )
                )

                # Convert gripper state to array format
                gripper_state = np.array([action_dict["gripper"]])

                # Create and send robot actions
                current_time = time.time()
                for i in range(self.control_interval):
                    action = RobotAction(
                        tcp_position=self._current_tcp_position,
                        tcp_orientation=self._current_tcp_orientation,
                        gripper_state=gripper_state,
                        control_mode="cartesian",
                        timestamp=current_time,
                        target_timestamp=current_time
                        + (i + 2) * self.dt
                        - self.command_latency,
                    )
                    self.shared_storage.write_action(action)

                # Check for errors
                if self.shared_storage.error_state.value:
                    break

                rate.sleep()

        except Exception as e:
            print(f"Error in QuestPolicy: {e}")
            traceback.print_exc()
            self.shared_storage.error_state.value = True

    def stop(self):
        """Stop the policy process."""
        # Stop recording if active
        if self.shared_storage.is_recording.value:
            self.shared_storage.is_recording.value = False
            if self._current_episode_dir:
                print("Recording stopped")

        if self._listener is not None:
            self._listener.stop()
        super().stop()
