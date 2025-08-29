#!/usr/bin/env python3
import signal
import sys
import time
from typing import Any, Dict, List
import numpy as np
from scipy.spatial.transform import Rotation as R
import torch.multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import hydra
from pynput import keyboard
from omegaconf import OmegaConf

from maniunicon.utils.shared_memory.shared_storage import SharedStorage


class WorkspaceBoundsRecorder:
    """Record workspace bounds by controlling robot to 8 boundary poses."""

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        robot_cfg: Dict[str, Any],
        policy_cfg: Dict[str, Any],
    ):
        # Create shared memory
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.shared_storage = SharedStorage(
            shm_manager=self.shm_manager,
            robot_state_config=data_cfg.robot_state_config,
            robot_action_config=data_cfg.robot_action_config,
            camera_config=None,  # No camera needed
        )
        self.reset_event = mp.Event()

        self.robot = hydra.utils.instantiate(
            robot_cfg,
            shared_storage=self.shared_storage,
            reset_event=self.reset_event,
        )

        self.policy = hydra.utils.instantiate(
            policy_cfg,
            shared_storage=self.shared_storage,
            reset_event=self.reset_event,
            _recursive_=False,
        )

        # Recording state
        self.current_pose_idx = 0
        self.recorded_poses = []
        self.recording_complete = False
        self.stopping = False
        self.pose_descriptions = [
            "LEFT-FRONT-TOP corner position",
            "RIGHT-BACK-BOTTOM corner position",
            "MAX X rotation",
            "MIN X rotation",
            "MAX Y rotation",
            "MIN Y rotation",
            "MAX Z rotation",
            "MIN Z rotation",
        ]

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start(self):
        """Start all processes."""
        print("Starting workspace bounds recorder...")
        print("Use your configured policy to control the robot")
        print("Press 'p' to save current pose")
        print("Press 'h' to reset robot to home position")
        print("Press Ctrl+C to exit")
        print("=" * 60)

        # Start processes
        self.policy.start()
        print("Policy started.")

        self.robot.start()
        print("Robot started.")

        # Set up keyboard listener for saving poses
        def _on_press(key):
            if key == keyboard.KeyCode.from_char("p"):
                self._save_current_pose()
            elif key == keyboard.KeyCode.from_char("h"):
                print("Resetting robot to home position...")
                self.reset_event.set()

        self.keyboard_listener = keyboard.Listener(on_press=_on_press)
        self.keyboard_listener.start()
        print("Keyboard listener started")

        # Display first pose instruction
        self._show_current_pose_instruction()

        print("All processes started")

    def _save_current_pose(self):
        """Save the current robot pose."""
        try:
            # Get current robot state
            state = self.shared_storage.read_state()
            if state is None:
                print("Error: Could not read robot state")
                return

            # Extract pose information
            tcp_position = state.tcp_position
            tcp_orientation = state.tcp_orientation

            # Convert quaternion to Euler angles using scipy Rotation
            # Note: robot interface returns quaternion in [x, y, z, w] format
            # scipy Rotation accepts this format directly
            rotation = R.from_quat(tcp_orientation)
            roll_val, pitch_val, yaw_val = rotation.as_euler("xyz", degrees=False)

            # Record the pose
            pose_data = {
                "description": self.pose_descriptions[self.current_pose_idx],
                "position": tcp_position.copy(),
                "orientation_quat": tcp_orientation.copy(),
                "roll": roll_val,
                "pitch": pitch_val,
                "yaw": yaw_val,
            }

            self.recorded_poses.append(pose_data)

            # Print what was saved (with newline to clear the current display)
            print()  # Clear the current pose display line
            if self.current_pose_idx < 2:
                # For position poses, print XYZ
                print(f"✓ Saved {self.pose_descriptions[self.current_pose_idx]}:")
                print(
                    f"  Position: [{tcp_position[0]:.4f}, {tcp_position[1]:.4f}, {tcp_position[2]:.4f}]"
                )
            else:
                # For rotation poses, print the specific rotation value
                rotation_type = self.pose_descriptions[self.current_pose_idx].split()[
                    1
                ]  # MAX/MIN
                axis = self.pose_descriptions[self.current_pose_idx].split()[2]  # X/Y/Z

                if axis == "X":
                    value = roll_val
                elif axis == "Y":
                    value = pitch_val
                else:  # Z
                    value = yaw_val

                print(f"✓ Saved {self.pose_descriptions[self.current_pose_idx]}:")
                print(f"  {axis}: {value:.4f} rad ({np.degrees(value):.1f}°)")

            # Move to next pose
            self.current_pose_idx += 1

            if self.current_pose_idx < len(self.pose_descriptions):
                print()
                self._show_current_pose_instruction()
            else:
                print()
                print("=" * 60)
                print("All poses recorded! Final summary:")
                self._print_final_summary()
                print("Stopping system...")
                self.recording_complete = True
                self.stop()

        except Exception as e:
            print(f"Error saving pose: {e}")
            import traceback

            traceback.print_exc()

    def _show_current_pose_instruction(self):
        """Show instruction for current pose to record."""
        print(
            f"[{self.current_pose_idx + 1}/8] Move robot to: {self.pose_descriptions[self.current_pose_idx]}"
        )
        print("Press 'p' to save this pose")
        print("-" * 40)

    def _print_current_pose_value(self):
        """Print the current pose value that would be saved."""
        try:
            # Get current robot state
            state = self.shared_storage.read_state()
            if state is None:
                print("Current: [No robot state available]")
                return

            # Extract pose information
            tcp_position = state.tcp_position
            tcp_orientation = state.tcp_orientation

            # Convert quaternion to Euler angles using scipy Rotation
            # Note: robot interface returns quaternion in [x, y, z, w] format
            # scipy Rotation accepts this format directly
            rotation = R.from_quat(tcp_orientation)
            roll_val, pitch_val, yaw_val = rotation.as_euler("xyz", degrees=False)

            # Print the relevant value based on current pose index
            if self.current_pose_idx < 2:
                # For position poses, print XYZ
                print(
                    f"\rCurrent Position: [{tcp_position[0]:.4f}, {tcp_position[1]:.4f}, {tcp_position[2]:.4f}]",
                    end="",
                    flush=True,
                )
            else:
                # For rotation poses, print the specific rotation value
                axis = self.pose_descriptions[self.current_pose_idx].split()[2]  # X/Y/Z

                if axis == "X":
                    value = roll_val
                elif axis == "Y":
                    value = pitch_val
                else:  # Z
                    value = yaw_val

                print(
                    f"\rCurrent {axis}: {value:.4f} rad ({np.degrees(value):.1f}°)",
                    end="",
                    flush=True,
                )

        except Exception as e:
            print(f"\rCurrent: [Error reading state: {e}]", end="", flush=True)

    def _print_final_summary(self):
        """Print final summary of all recorded bounds."""
        print()
        print("WORKSPACE BOUNDS SUMMARY:")
        print("=" * 50)

        # Position bounds
        pos1 = self.recorded_poses[0]["position"]
        pos2 = self.recorded_poses[1]["position"]

        x_min, x_max = min(pos1[0], pos2[0]), max(pos1[0], pos2[0])
        y_min, y_max = min(pos1[1], pos2[1]), max(pos1[1], pos2[1])
        z_min, z_max = min(pos1[2], pos2[2]), max(pos1[2], pos2[2])

        print("Position Bounds:")
        print(f"  X: [{x_min:.4f}, {x_max:.4f}] (range: {x_max - x_min:.4f})")
        print(f"  Y: [{y_min:.4f}, {y_max:.4f}] (range: {y_max - y_min:.4f})")
        print(f"  Z: [{z_min:.4f}, {z_max:.4f}] (range: {z_max - z_min:.4f})")

        # Orientation bounds
        print("\nOrientation Bounds:")

        # X bounds
        roll_max = self.recorded_poses[2]["roll"]
        roll_min = self.recorded_poses[3]["roll"]
        print(
            f"  X: [{roll_min:.4f}, {roll_max:.4f}] rad "
            f"([{np.degrees(roll_min):.1f}°, {np.degrees(roll_max):.1f}°])"
        )

        # Y bounds
        pitch_max = self.recorded_poses[4]["pitch"]
        pitch_min = self.recorded_poses[5]["pitch"]
        print(
            f"  Y: [{pitch_min:.4f}, {pitch_max:.4f}] rad "
            f"([{np.degrees(pitch_min):.1f}°, {np.degrees(pitch_max):.1f}°])"
        )

        # Z bounds
        yaw_max = self.recorded_poses[6]["yaw"]
        yaw_min = self.recorded_poses[7]["yaw"]
        print(
            f"  Z: [{yaw_min:.4f}, {yaw_max:.4f}] rad "
            f"([{np.degrees(yaw_min):.1f}°, {np.degrees(yaw_max):.1f}°])"
        )

        print("\nAll poses recorded successfully!")
        print("=" * 50)

    def stop(self):
        """Stop all processes gracefully."""
        if self.stopping:
            return  # Already stopping, avoid duplicate messages

        self.stopping = True
        print("Stopping workspace bounds recorder...")

        # Stop keyboard listener
        if hasattr(self, "keyboard_listener"):
            self.keyboard_listener.stop()

        # Stop processes in reverse order
        self.robot.disconnect()
        self.robot.stop()
        self.policy.stop()

        self.shared_storage.is_running.value = False
        self.shm_manager.shutdown()

        print("All processes stopped")

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        print(f"Received signal {signum}")
        self.stop()
        sys.exit(0)


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="default",
)
def main(cfg):
    """Main entry point for the workspace bounds recorder."""
    # Register custom OmegaConf resolver for mathematical expressions
    OmegaConf.register_new_resolver("eval", eval)

    print(f"Configuration: {cfg}")
    mp.set_start_method("spawn")

    # Create and start recorder
    recorder = WorkspaceBoundsRecorder(
        data_cfg=cfg.data,
        robot_cfg=cfg.robot,
        policy_cfg=cfg.policy,
    )

    try:
        recorder.start()

        # Keep main thread alive and continuously display current pose
        last_print_time = 0
        print_interval = 0.5  # Print every 0.5 seconds

        while (
            recorder.current_pose_idx < len(recorder.pose_descriptions)
            and not recorder.recording_complete
        ):
            current_time = time.time()

            if current_time - last_print_time >= print_interval:
                recorder._print_current_pose_value()
                last_print_time = current_time

            time.sleep(0.1)

        # Wait a bit for stop() to complete, then exit
        if recorder.recording_complete:
            time.sleep(1)  # Give stop() time to complete
            print("Exiting...")
            return
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        import traceback

        print(f"Error in main: {e}")
        traceback.print_exc()
        recorder.stop()
    finally:
        recorder.stop()


if __name__ == "__main__":
    main()
