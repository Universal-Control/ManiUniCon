import os
import traceback
import numpy as np
from typing import Dict, Any, Optional
from loop_rate_limiters import RateLimiter

from maniunicon.core.sensor import BaseSensor
from maniunicon.utils.shared_memory.shared_storage import (
    RobotState,
    MultiCameraData,
)
from maniunicon.utils.replay_buffer import ReplayBuffer


class ReplaySensor(BaseSensor):
    """Sensor that replays recorded data from zarr files."""

    def __init__(
        self,
        shared_storage,
        data_file: str,
        frequency: float = 30.0,
        loop: bool = True,
        start_from_beginning: bool = True,
        load_to_memory: bool = False,
        name: str = "ReplaySensor",
    ):
        """Initialize the ReplaySensor.

        Args:
            shared_storage: SharedStorage instance
            data_file: Path to the zarr file containing recorded data
            frequency: Frequency to replay data in Hz
            loop: Whether to loop the data when reaching the end
            start_from_beginning: Whether to start from the beginning of the data
            name: Process name
        """
        super().__init__(shared_storage, frequency, name)
        self.data_file = data_file
        self.loop = loop
        self.start_from_beginning = start_from_beginning
        self.load_to_memory = load_to_memory
        self.replay_buffer = None
        self.current_step = 0
        self.data_loaded = False

    def _load_data(self):
        """Load data from the zarr file."""
        try:
            if not os.path.exists(self.data_file):
                raise FileNotFoundError(f"Data file not found: {self.data_file}")

            # Load the replay buffer from zarr file
            if self.load_to_memory:
                self.replay_buffer = ReplayBuffer.copy_from_path(
                    zarr_path=self.data_file, mode="r"
                )
            else:
                self.replay_buffer = ReplayBuffer.create_from_path(
                    zarr_path=self.data_file, mode="r"
                )

            if self.replay_buffer.n_steps == 0:
                raise ValueError("Zarr file contains no data")

            print(
                f"[ReplaySensor] Loaded {self.replay_buffer.n_steps} steps from {self.data_file}"
            )
            print(f"[ReplaySensor] Contains {self.replay_buffer.n_episodes} episodes")
            self.data_loaded = True

            # Reset step if starting from beginning
            if self.start_from_beginning:
                self.current_step = 0

        except Exception as e:
            print(f"[ReplaySensor] Error loading data: {e}")
            self.shared_storage.error_state.value = True
            raise

    def _convert_to_robot_state(self, obs_data: Dict[str, Any]) -> Optional[RobotState]:
        """Convert observation data to RobotState if it contains state data."""
        try:
            # Check if this observation contains robot state data
            # We need at least joint_positions to consider it robot state data
            if "joint_positions" not in obs_data:
                return None

            joint_positions = obs_data["joint_positions"][0]
            joint_velocities = obs_data["joint_velocities"][0]
            joint_torques = obs_data["joint_torques"][0]
            tcp_position = obs_data["tcp_position"][0]
            tcp_orientation = obs_data["tcp_orientation"][0]
            timestamp = obs_data["timestamp"][0]

            return RobotState(
                joint_positions=joint_positions,
                joint_velocities=joint_velocities,
                joint_torques=joint_torques,
                tcp_position=tcp_position,
                tcp_orientation=tcp_orientation,
                gripper_state=obs_data.get("gripper_state", np.array([0.0])),
                timestamp=np.array(timestamp),
            )
        except Exception as e:
            print(f"[ReplaySensor] Error converting to RobotState: {e}")
            traceback.print_exc()

        return None

    def _convert_to_multi_camera_data(
        self, obs_data: Dict[str, Any]
    ) -> Optional[MultiCameraData]:
        """Convert observation data to MultiCameraData if it contains multi-camera data."""
        try:
            # Check if this observation contains multi-camera data
            if "images" not in obs_data and "depths" not in obs_data:
                return None

            # Extract camera data
            colors = None
            depths = None

            # Convert images dict to list format expected by MultiCameraData
            camera_names = sorted(obs_data["images"].keys())
            colors = np.concatenate(
                [obs_data["images"][cam] for cam in camera_names], axis=0
            )

            # Convert depths dict to list format
            depths = np.concatenate(
                [obs_data["depths"][cam] for cam in camera_names], axis=0
            )
            # remove the potential last dimension of depth map
            depths = depths.reshape(depths.shape[0], depths.shape[1], depths.shape[2])

            return MultiCameraData(
                depths=depths,
                colors=colors,
                intrs=obs_data["intrs"][0],
                transforms=obs_data["transforms"][0],
                timestamp=np.array(obs_data["timestamp"][0]),
            )

        except Exception as e:
            print(f"[ReplaySensor] Error converting to MultiCameraData: {e}")
            traceback.print_exc()

        return None

    def _replay_data_step(self, step_idx: int):
        """Replay a single data step to shared storage."""
        try:
            data_written = False

            # Get observation data for this step using ReplayBuffer's method
            step_data = self.replay_buffer.get_steps_slice(step_idx, step_idx + 1)
            step_obs = step_data["obs"]

            # Try to convert and write as robot state
            robot_state = self._convert_to_robot_state(step_obs)
            if robot_state is not None:
                self.shared_storage.write_state(robot_state)
                data_written = True

            # Try to convert and write as multi-camera data
            multi_camera_data = self._convert_to_multi_camera_data(step_obs)
            if multi_camera_data is not None:
                if (
                    hasattr(self.shared_storage, "multi_camera_buffer")
                    and self.shared_storage.multi_camera_buffer is not None
                ):
                    self.shared_storage.write_multi_camera(multi_camera_data)
                    data_written = True

            if not data_written:
                # Log available fields for debugging
                available_fields = list(step_obs.keys())
                print(
                    f"[ReplaySensor] Warning: Could not identify data type for step {step_idx}"
                )
                print(f"[ReplaySensor] Available fields: {available_fields}")

        except Exception as e:
            print(f"[ReplaySensor] Error replaying data step: {e}")
            traceback.print_exc()

    def run(self):
        """Main loop that replays data at specified frequency."""
        try:
            # Load data first
            self._load_data()

            if not self.data_loaded:
                print("[ReplaySensor] Failed to load data, exiting")
                return

            print(f"[ReplaySensor] Starting replay at {self.frequency} Hz")
            print(f"[ReplaySensor] Loop mode: {'enabled' if self.loop else 'disabled'}")

            rate = RateLimiter(
                frequency=self.frequency, warn=True, name="replay_sensor"
            )
            while self.shared_storage.is_running.value:
                try:
                    # Check if we've reached the end of data
                    if self.current_step >= self.replay_buffer.n_steps:
                        if self.loop:
                            print(
                                "[ReplaySensor] Reached end of data, looping back to beginning"
                            )
                            self.current_step = 0
                        else:
                            print("[ReplaySensor] Reached end of data, stopping replay")
                            break

                    # Replay the current step
                    self._replay_data_step(self.current_step)

                    # Move to next step
                    self.current_step += 1

                    # Log progress periodically
                    if self.current_step % 100 == 0:
                        progress = (
                            self.current_step / self.replay_buffer.n_steps
                        ) * 100
                        print(
                            f"[ReplaySensor] Progress: {self.current_step}/{self.replay_buffer.n_steps} ({progress:.1f}%)"
                        )

                except Exception as e:
                    print(f"[ReplaySensor] Error in replay loop: {e}")
                    self.shared_storage.error_state.value = True
                    break

                rate.sleep()

        except Exception as e:
            print(f"[ReplaySensor] Fatal error: {e}")
            self.shared_storage.error_state.value = True
        finally:
            print("[ReplaySensor] Replay finished")

    def get_progress(self) -> Dict[str, Any]:
        """Get current replay progress information."""
        if not self.data_loaded:
            return {"loaded": False}

        return {
            "loaded": True,
            "total_steps": self.replay_buffer.n_steps,
            "current_step": self.current_step,
            "progress_percent": (
                (self.current_step / self.replay_buffer.n_steps) * 100
                if self.replay_buffer.n_steps > 0
                else 0
            ),
            "n_episodes": self.replay_buffer.n_episodes,
            "loop_enabled": self.loop,
        }

    def seek(self, step: int):
        """Seek to a specific step in the data."""
        if not self.data_loaded:
            print("[ReplaySensor] Data not loaded, cannot seek")
            return

        if 0 <= step < self.replay_buffer.n_steps:
            self.current_step = step
            print(f"[ReplaySensor] Seeked to step {step}")
        else:
            print(
                f"[ReplaySensor] Invalid seek step {step}, valid range: 0-{self.replay_buffer.n_steps-1}"
            )

    def stop(self):
        """Stop the replay sensor."""
        print("[ReplaySensor] Stopping...")
        self.shared_storage.is_running.value = False
        self.join()
