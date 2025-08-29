#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Keyboard-based teleoperation policy for robot control."""

import os
import time
from typing import Optional
import numpy as np
from pynput import keyboard
from multiprocessing.synchronize import Event
import traceback
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation

from maniunicon.utils.shared_memory.shared_storage import (
    SharedStorage,
    RobotAction,
    RobotState,
)
from maniunicon.core.policy import BasePolicy
from maniunicon.utils.data import get_next_episode_dir


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two quaternions q1 * q2 in [x, y, z, w] order.

    Args:
        q1: First quaternion [x, y, z, w]
        q2: Second quaternion [x, y, z, w]

    Returns:
        Quaternion product [x, y, z, w]
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([x, y, z, w])


def exp3(v: np.ndarray) -> np.ndarray:
    """Convert rotation vector to quaternion using exponential map, output [x, y, z, w].

    Args:
        v: Rotation vector [x, y, z] in radians

    Returns:
        Quaternion [x, y, z, w]
    """
    theta = np.linalg.norm(v)
    if theta < 1e-7:
        return np.array([0.0, 0.0, 0.0, 1.0])

    # Normalize rotation vector
    v = v / theta

    # Compute quaternion components
    w = np.cos(theta / 2.0)
    xyz = v * np.sin(theta / 2.0)

    return np.array([xyz[0], xyz[1], xyz[2], w])


class KeyboardPolicy(BasePolicy):
    """Keyboard-based teleoperation policy.

    Controls:
    - WASD: Move end-effector in XY plane
    - QE: Move end-effector up/down (Z)
    - Arrow keys: Rotate end-effector around X/Y axes
    - ZC: Rotate end-effector around Z axis
    - Space: Emergency stop
    - Esc: Disconnect
    - 1-7: Direct joint control (hold key to move joint)
    - 'b': Reverse joint movement
    - 'r': Toggle recording (start/stop episode recording)
    - 'g': Toggle gripper open/close
    - 'h': Reset robot to home position
    """

    def __init__(
        self,
        shared_storage: SharedStorage,
        reset_event: Event,
        position_step: float = 0.01,  # meters
        orientation_step: float = 0.1,  # radians
        joint_step: float = 0.1,  # radians
        num_joints: int = 7,  # Number of robot joints
        control_interval: int = 1,  # Number of action steps to send
        dt: float = 0.01,  # Time step between actions
        command_latency: float = 0.01,  # seconds
        record_dir: Optional[str] = None,  # Directory to save recordings
        synchronized: bool = False,
        warn_on_late: bool = True,
        workspace_bounds: dict = None,
        name: str = "KeyboardPolicy",
    ):
        super().__init__(
            shared_storage=shared_storage,
            reset_event=reset_event,
            command_latency=command_latency,
            name=name,
        )
        self.position_step = position_step
        self.orientation_step = orientation_step
        self.joint_step = joint_step
        self.num_joints = num_joints
        self.control_interval = control_interval
        self.dt = dt
        self.frequency = 1 / dt
        self.record_dir = record_dir
        self.synchronized = synchronized
        self.warn_on_late = warn_on_late
        self.workspace_bounds = workspace_bounds

        # Internal state
        self._listener: Optional[keyboard.Listener] = None
        self._pressed_keys = set()
        self._gripper_state = np.array([0.0])  # 0 for open, 1 for closed
        self._gripper_key_pressed = False
        self._recording_key_pressed = False
        self._recording_active = False
        self._current_episode_dir: Optional[str] = None
        self._emergency_stop = False
        self._should_disconnect = False

        # Cached robot state
        self._current_tcp_position = None
        self._current_tcp_orientation = None
        self._current_joint_positions = None
        self._control_mode = "cartesian"

        # Key mappings
        self.position_keys = {
            keyboard.KeyCode.from_char("w"): np.array(
                [self.position_step, 0.0, 0.0]
            ),  # Forward (X+)
            keyboard.KeyCode.from_char("s"): np.array(
                [-self.position_step, 0.0, 0.0]
            ),  # Backward (X-)
            keyboard.KeyCode.from_char("a"): np.array(
                [0.0, self.position_step, 0.0]
            ),  # Left (Y+)
            keyboard.KeyCode.from_char("d"): np.array(
                [0.0, -self.position_step, 0.0]
            ),  # Right (Y-)
            keyboard.KeyCode.from_char("q"): np.array(
                [0.0, 0.0, self.position_step]
            ),  # Up (Z+)
            keyboard.KeyCode.from_char("e"): np.array(
                [0.0, 0.0, -self.position_step]
            ),  # Down (Z-)
        }

        self.orientation_keys = {
            keyboard.Key.up: np.array(
                [0.0, self.orientation_step, 0.0]
            ),  # Rotate around Y
            keyboard.Key.down: np.array(
                [0.0, -self.orientation_step, 0.0]
            ),  # Rotate around Y
            keyboard.Key.left: np.array(
                [self.orientation_step, 0.0, 0.0]
            ),  # Rotate around X
            keyboard.Key.right: np.array(
                [-self.orientation_step, 0.0, 0.0]
            ),  # Rotate around X
            keyboard.KeyCode.from_char("z"): np.array(
                [0.0, 0.0, self.orientation_step]
            ),  # Rotate around Z
            keyboard.KeyCode.from_char("c"): np.array(
                [0.0, 0.0, -self.orientation_step]
            ),  # Rotate around Z
        }

        # Joint keys for n-DoF robot
        self.joint_keys = {
            keyboard.KeyCode.from_char(f"{i + 1}"): i for i in range(self.num_joints)
        }

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
            rotation = Rotation.from_quat(tcp_orientation)
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
            clipped_rotation = Rotation.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False
            )
            clipped_orientation = clipped_rotation.as_quat()

        return clipped_position, clipped_orientation

    def _on_press(self, key):
        """Handle key press events."""
        try:
            if key not in self._pressed_keys:
                self._pressed_keys.add(key)

                # Handle recording toggle on key press (not hold)
                if (
                    key == keyboard.KeyCode.from_char("r")
                    and not self._recording_key_pressed
                ):
                    self._recording_key_pressed = True
                    self._recording_active = not self._recording_active
                    print(f"Recording: {'ON' if self._recording_active else 'OFF'}")

                # Handle gripper toggle on key press (not hold)
                if (
                    key == keyboard.KeyCode.from_char("g")
                    and not self._gripper_key_pressed
                ):
                    self._gripper_key_pressed = True
                    self._gripper_state = 1.0 - self._gripper_state
                    print(
                        f"Gripper: {'CLOSE' if self._gripper_state[0] > 0 else 'OPEN'}"
                    )

                # Handle emergency stop
                if key == keyboard.Key.space:
                    self._emergency_stop = True
                    print("EMERGENCY STOP")

                # Handle disconnect
                if key == keyboard.Key.esc:
                    self._should_disconnect = True

        except AttributeError:
            pass

    def _on_release(self, key):
        """Handle key release events."""
        try:
            if key in self._pressed_keys:
                self._pressed_keys.remove(key)

            # Reset recording key press state
            if key == keyboard.KeyCode.from_char("r"):
                self._recording_key_pressed = False

            # Reset gripper key press state
            if key == keyboard.KeyCode.from_char("g"):
                self._gripper_key_pressed = False

            # Reset emergency stop
            if key == keyboard.Key.space:
                self._emergency_stop = False

        except AttributeError:
            pass

    def _update_control_mode(self):
        """Update control mode based on pressed keys."""
        for key in self._pressed_keys:
            if key in self.joint_keys:
                if self._control_mode != "joint":
                    self.sync_state()
                self._control_mode = "joint"
                return
        # Default to cartesian if no joint keys pressed
        for key in self._pressed_keys:
            if key in self.position_keys or key in self.orientation_keys:
                if self._control_mode != "cartesian":
                    self.sync_state()
                self._control_mode = "cartesian"
                return

    def sync_state(self):
        """Sync the state of the robot with the shared storage."""
        state = None
        while state is None:
            state: RobotState | None = self.shared_storage.read_state(k=1)
            print("KeyboardPolicy: waiting for state...")
            time.sleep(0.05)

        self._current_tcp_position = state.tcp_position[0].copy()
        self._current_tcp_orientation = state.tcp_orientation[0].copy()
        self._current_joint_positions = state.joint_positions[0].copy()

    def run(self):
        """Main process loop."""
        try:
            # Start keyboard listener
            self._listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self._listener.start()
            print("Keyboard controller connected successfully")
            print("\nKeyboard controls:")
            print("WASD: Move end-effector in XY plane")
            print("QE: Move end-effector up/down (Z)")
            print("Arrow keys: Rotate end-effector around X/Y axes")
            print("ZC: Rotate end-effector around Z axis")
            print(f"1-{self.num_joints}: Direct joint control (hold key to move joint)")
            print("b + 1-N: Reverse joint movement")
            print("g: Toggle gripper open/close")
            print("r: Toggle recording (start/stop episode recording)")
            print("h: Reset robot to home position")
            print("Space: Emergency stop")
            print("Esc: Disconnect")
            if self.record_dir is not None:
                print(f"\nRecording directory: {self.record_dir}")
            self.sync_state()

            rate = RateLimiter(
                frequency=self.frequency,
                warn=self.warn_on_late,
                name="keyboard_policy",
            )
            while self.shared_storage.is_running.value and not self._should_disconnect:
                # Handle reset
                if self.reset_event is not None and self.reset_event.is_set():
                    # Clear actions during reset
                    self.shared_storage.read_all_action()
                    self.sync_state()
                    rate.sleep()
                    continue

                if self.synchronized and (
                    self.reset_event is None or not self.reset_event.is_set()
                ):
                    # wait for robot to be ready
                    self.shared_storage.robot_ready.wait()
                    self.shared_storage.robot_ready.clear()

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
                            start_time=time.time() + 2 * self.dt, dt=self.dt
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

                # Update control mode
                self._update_control_mode()

                # Process position keys
                for key, delta in self.position_keys.items():
                    if key in self._pressed_keys:
                        self._current_tcp_position += delta

                # Process orientation keys
                for key, delta in self.orientation_keys.items():
                    if key in self._pressed_keys:
                        # Convert delta to quaternion using exponential map
                        delta_quat = exp3(delta)
                        # Multiply with current orientation
                        self._current_tcp_orientation = quat_multiply(
                            delta_quat, self._current_tcp_orientation
                        )
                        # Normalize quaternion, but not in place
                        self._current_tcp_orientation /= np.linalg.norm(
                            self._current_tcp_orientation
                        )

                # Apply workspace bounds clipping to internal state
                self._current_tcp_position, self._current_tcp_orientation = (
                    self._clip_tcp_pose_to_bounds(
                        self._current_tcp_position, self._current_tcp_orientation
                    )
                )

                # Process joint keys
                reverse_pressed = keyboard.KeyCode.from_char("b") in self._pressed_keys
                for key, joint_idx in self.joint_keys.items():
                    if joint_idx < self.num_joints and key in self._pressed_keys:
                        direction = -1 if reverse_pressed else 1
                        self._current_joint_positions[joint_idx] += (
                            direction * self.joint_step
                        )

                # Create and send robot actions
                current_time = time.time()
                for i in range(self.control_interval):
                    if self._control_mode == "cartesian":
                        action = RobotAction(
                            tcp_position=self._current_tcp_position,
                            tcp_orientation=self._current_tcp_orientation,
                            gripper_state=self._gripper_state,
                            control_mode="cartesian",
                            timestamp=current_time,
                            target_timestamp=current_time
                            + (i + 2) * self.dt
                            - self.command_latency,
                        )
                    else:  # joint mode
                        action = RobotAction(
                            joint_positions=self._current_joint_positions,
                            gripper_state=self._gripper_state,
                            control_mode="joint",
                            timestamp=current_time,
                            target_timestamp=current_time
                            + (i + 2) * self.dt
                            - self.command_latency,
                        )

                    # Override with emergency stop if active
                    if self._emergency_stop:
                        action.control_mode = "joint"
                        action.joint_positions = self._current_joint_positions
                        action.tcp_position = None
                        action.tcp_orientation = None

                    self.shared_storage.write_action(action)

                if self.synchronized and (
                    self.reset_event is None or not self.reset_event.is_set()
                ):
                    self.shared_storage.policy_ready.set()  # Signal robot can execute actions

                # Check for errors
                if self.shared_storage.error_state.value:
                    break

                if not self.synchronized:
                    rate.sleep()

        except Exception as e:
            print(f"Error in KeyboardPolicy: {e}")
            traceback.print_exc()
            self.shared_storage.error_state.value = True
        finally:
            if self._listener is not None:
                self._listener.stop()

    def stop(self):
        """Stop the policy process."""
        # Stop recording if active
        if self.shared_storage.is_recording.value:
            self.shared_storage.is_recording.value = False
            if self._current_episode_dir:
                print(
                    f"Recording stopped - Episode saved to: {self._current_episode_dir}"
                )

        if self._listener is not None:
            self._listener.stop()
        super().stop()
