#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""SpaceMouse-based teleoperation policy for robot control."""

import os
import time
import threading
from typing import Optional
from collections import namedtuple
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

try:
    import hid
except ModuleNotFoundError as e:
    raise ImportError(
        "Unable to load module hid, required to interface with SpaceMouse. "
        "`pip install hidapi` to install the additional requirements."
    ) from e


# axis mappings are specified as:
# [channel, byte1, byte2, scale]; scale is usually just -1 or 1 and multiplies the result by this value
AxisSpec = namedtuple("AxisSpec", ["channel", "byte1", "byte2", "scale"])

# button states are specified as:
# [channel, data byte,  bit of byte, index to write to]
ButtonSpec = namedtuple("ButtonSpec", ["channel", "byte", "bit"])


def to_int16(y1, y2):
    """Convert two 8 bit bytes to a signed 16 bit integer."""
    x = (y1) | (y2 << 8)
    if x >= 32768:
        x = -(65536 - x)
    return x


def clip_delta_min(delta: np.ndarray, threshold: float) -> np.ndarray:
    """Clip small deltas to zero to reduce noise."""
    return np.where(np.abs(delta) < threshold, 0, delta)


class DeviceSpec:
    def __init__(
        self, vendor_id, product_id, axis_mapping, button_mapping, axis_scale, max_bytes
    ):
        self.vendor_id = vendor_id
        self.product_id = product_id
        self.axis_mapping = axis_mapping
        self.button_mapping = button_mapping
        self.axis_scale = axis_scale
        self.max_bytes = max_bytes


class SpaceMouseData:
    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.left = False
        self.right = False


class SpaceMouseDevice:
    """Low-level SpaceMouse device interface."""

    def __init__(self):
        self.device = None
        self.device_spec = None

    def open(self, device_spec: DeviceSpec):
        if self.device is not None:
            self.close()
        self.device = hid.device()
        self.device.open(device_spec.vendor_id, device_spec.product_id)
        print("Manufacturer: {:s}".format(self.device.get_manufacturer_string()))
        self.device_spec = device_spec

    def close(self):
        if self.device is not None:
            self.device.close()
            print(
                "Closed device: ({}, {})".format(
                    self.device_spec.vendor_id, self.device_spec.product_id
                )
            )
        self.device = None

    def __del__(self):
        self.close()

    def read(self) -> list[int]:
        """Read raw bytes from the SpaceMouse."""
        if self.device is None:
            raise RuntimeError("Device not open")
        return self.device.read(self.device_spec.max_bytes)

    def decode(self, data) -> dict:
        """Decode the data received from the SpaceMouse."""
        ret = {}

        for name, axis_spec in self.device_spec.axis_mapping.items():
            channel, byte1, byte2, scale = axis_spec
            if data[0] == channel:
                axis_value = to_int16(data[byte1], data[byte2])
                ret[name] = axis_value * scale / float(self.device_spec.axis_scale)

        for name, button_spec in self.device_spec.button_mapping.items():
            channel, byte, bit = button_spec
            if data[0] == channel:
                mask = 1 << bit
                ret[name] = (data[byte] & mask) != 0

        return ret


def parse_device_specs(specs_dict):
    """Parse device specifications from config dictionary."""
    device_specs = {}
    for name, spec in specs_dict.items():
        axis_mapping = {}
        for axis_name, axis_data in spec["axis_mapping"].items():
            axis_mapping[axis_name] = AxisSpec(
                channel=axis_data["channel"],
                byte1=axis_data["byte1"],
                byte2=axis_data["byte2"],
                scale=axis_data["scale"],
            )

        button_mapping = {}
        for button_name, button_data in spec["button_mapping"].items():
            button_mapping[button_name] = ButtonSpec(
                channel=button_data["channel"],
                byte=button_data["byte"],
                bit=button_data["bit"],
            )

        device_specs[name] = DeviceSpec(
            vendor_id=spec["vendor_id"],
            product_id=spec["product_id"],
            axis_mapping=axis_mapping,
            button_mapping=button_mapping,
            axis_scale=spec["axis_scale"],
            max_bytes=spec["max_bytes"],
        )
    return device_specs


def get_available_devices(device_specs):
    """Find available SpaceMouse devices."""
    available_devices = []
    for info in hid.enumerate():
        for spec in device_specs.values():
            if (
                info["vendor_id"] == spec.vendor_id
                and info["product_id"] == spec.product_id
            ):
                print(f"Found device: {info}")
                available_devices.append(spec)
                break
    return available_devices


class SpaceMouseThread:
    """A daemon thread to listen to SpaceMouse."""

    def __init__(
        self,
        device_spec: DeviceSpec | None = None,
        device_specs_dict: dict | None = None,
    ):
        if device_spec is None:
            if device_specs_dict is None:
                raise RuntimeError(
                    "Either device_spec or device_specs_dict must be provided"
                )
            device_specs = parse_device_specs(device_specs_dict)
            available = get_available_devices(device_specs)
            if len(available) == 0:
                raise RuntimeError("SpaceMouse not found")
            device_spec = available[0]

        self.device = SpaceMouseDevice()
        self.device.open(device_spec)
        self.state = SpaceMouseData()
        self._running = True
        self._lock = threading.Lock()

        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()

    def run(self):
        """A daemon thread to listen to SpaceMouse."""
        try:
            while self._running:
                data = self.device.read()
                state = self.device.decode(data)
                with self._lock:
                    for key, value in state.items():
                        setattr(self.state, key, value)
        except OSError:
            print("Fail to read from SpaceMouse")

    def close(self):
        self._running = False
        self.thread.join(timeout=1.0)
        self.device.close()

    def get_xyz(self):
        with self._lock:
            return np.array([self.state.x, self.state.y, self.state.z])

    def get_rpy(self):
        with self._lock:
            return np.array([self.state.roll, self.state.pitch, self.state.yaw])

    def get_left_button(self):
        with self._lock:
            return self.state.left

    def get_right_button(self):
        with self._lock:
            return self.state.right


class SpaceMousePolicy(BasePolicy):
    """SpaceMouse-based teleoperation policy.

    Controls:
    - Translation: Move end-effector in 3D space (X, Y, Z axes)
    - Rotation: Rotate end-effector around 3D axes (when rotation button is pressed)
    - Button 0 (Left): Toggle gripper open/close
    - Button 1 (Right): Enable rotation mode
    - 'r' key: Toggle recording (start/stop episode recording)
    - 'd' key: Drop current episode without saving
    """

    def __init__(
        self,
        shared_storage: SharedStorage,
        reset_event: Event,
        translation_scale: float = 1.5,  # mm
        rotation_scale: float = 1.2,  # degrees
        control_interval: int = 1,  # Number of action steps to send
        dt: float = 0.01,  # Time step between actions
        command_latency: float = 0.005,  # seconds
        record_dir: Optional[str] = None,
        synchronized: bool = False,
        warn_on_late: bool = True,
        workspace_bounds: dict = None,
        device_specs: dict = None,  # Device specifications from config
        name: str = "SpaceMousePolicy",
    ):
        super().__init__(
            shared_storage=shared_storage,
            reset_event=reset_event,
            command_latency=command_latency,
            name=name,
        )
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self.control_interval = control_interval
        self.dt = dt
        self.frequency = 1 / dt
        self.record_dir = record_dir
        self.synchronized = synchronized
        self.warn_on_late = warn_on_late
        self.workspace_bounds = workspace_bounds
        self.device_specs = device_specs

        # Internal state
        self._spacemouse_thread: Optional[SpaceMouseThread] = None
        self._gripper_state = np.array([0.0])
        self._last_left_button = False
        self._last_right_button = False
        self._recording_active = False
        self._recording_key_pressed = False
        self._current_episode_dir: Optional[str] = None
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
            print("SpaceMouse: waiting for state...")
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
            # Handle recording toggle on key press (not hold)
            if (
                key == keyboard.KeyCode.from_char("r")
                and not self._recording_key_pressed
            ):
                self._recording_key_pressed = True
                self._recording_active = not self._recording_active
                print(f"Recording: {'ON' if self._recording_active else 'OFF'}")
            # Handle dropping current episode
            elif key == keyboard.KeyCode.from_char("d") and self._recording_active:
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
        except AttributeError:
            pass

    def _on_release(self, key):
        """Handle key release events."""
        try:
            # Reset recording key press state
            if key == keyboard.KeyCode.from_char("r"):
                self._recording_key_pressed = False
        except AttributeError:
            pass

    def run(self):
        """Main process loop."""
        try:
            # Connect to SpaceMouse
            try:
                self._spacemouse_thread = SpaceMouseThread(
                    device_specs_dict=self.device_specs
                )
                print("SpaceMouse connected successfully")
                print("\nSpaceMouse controls:")
                print("Translation: Move end-effector in 3D space (X, Y, Z axes)")
                print(
                    "Rotation: Rotate end-effector around 3D axes (when rotation button is pressed)"
                )
                print("Button 0 (Left): Toggle gripper open/close")
                print("Button 1 (Right): Enable rotation mode")
                print("'r' key: Toggle recording (start/stop episode recording)")
                print("'d' key: Drop current episode without saving")
                if self.record_dir is not None:
                    print(f"\nRecording directory: {self.record_dir}")
                self.sync_state()
            except Exception as e:
                print(f"Failed to connect SpaceMouse: {e}")
                traceback.print_exc()
                self.shared_storage.error_state.value = True
                return

            # Start keyboard listener
            self._listener = keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self._listener.start()

            rate = RateLimiter(
                frequency=self.frequency,
                warn=self.warn_on_late,
                name="spacemouse_policy",
            )
            while self.shared_storage.is_running.value:
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

                # Get SpaceMouse data
                d_xyz = self._spacemouse_thread.get_xyz()
                d_rpy = self._spacemouse_thread.get_rpy()
                left_button = self._spacemouse_thread.get_left_button()
                right_button = self._spacemouse_thread.get_right_button()

                # Process translation
                delta_translation = np.array(
                    [
                        d_xyz[0] * self.translation_scale,
                        d_xyz[1] * self.translation_scale,
                        d_xyz[2] * self.translation_scale,
                    ]
                )

                # Apply threshold to reduce noise
                delta_translation = clip_delta_min(delta_translation, 0.001)

                # Update position
                self._current_tcp_position = (
                    self._current_tcp_position + delta_translation
                )

                # Process rotation (only when right button is pressed)
                if right_button:
                    # Apply threshold to rotation
                    d_rpy = clip_delta_min(d_rpy, 0.35)

                    # Convert SpaceMouse rotation to rotation matrix
                    rotation_delta_deg = d_rpy * self.rotation_scale

                    if np.any(np.abs(rotation_delta_deg) > 0):
                        rotation_delta = R.from_euler(
                            "zxy", rotation_delta_deg, degrees=True
                        )
                        current_rotation = R.from_quat(self._current_tcp_orientation)
                        # Apply rotation in tool frame
                        new_rotation = current_rotation * rotation_delta
                        self._current_tcp_orientation = new_rotation.as_quat()

                # Apply workspace bounds clipping to internal state
                self._current_tcp_position, self._current_tcp_orientation = (
                    self._clip_tcp_pose_to_bounds(
                        self._current_tcp_position, self._current_tcp_orientation
                    )
                )

                # Handle left button for gripper toggle
                if left_button and not self._last_left_button and not right_button:
                    self._gripper_state = 1.0 - self._gripper_state
                    print(
                        f"Gripper: {'CLOSE' if self._gripper_state[0] > 0 else 'OPEN'}"
                    )

                # Create and send robot actions
                current_time = time.time()
                for i in range(self.control_interval):
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
                    self.shared_storage.write_action(action)

                # Update button states
                self._last_left_button = left_button
                self._last_right_button = right_button

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
            print(f"Error in SpaceMousePolicy: {e}")
            traceback.print_exc()
            self.shared_storage.error_state.value = True
        finally:
            if self._spacemouse_thread is not None:
                self._spacemouse_thread.close()
            if self._listener is not None:
                self._listener.stop()

    def stop(self):
        """Stop the policy process."""
        # Stop recording if active
        if self.shared_storage.is_recording.value:
            self.shared_storage.is_recording.value = False
            if self._current_episode_dir:
                print("Recording stopped")

        if self._spacemouse_thread is not None:
            self._spacemouse_thread.close()
        if self._listener is not None:
            self._listener.stop()
        super().stop()
