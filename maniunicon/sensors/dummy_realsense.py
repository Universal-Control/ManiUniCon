import time
import traceback
from typing import Dict, Optional

import numpy as np
import cv2
from loop_rate_limiters import RateLimiter

from maniunicon.core.sensor import BaseSensor
from maniunicon.utils.shared_memory.shared_storage import (
    SharedStorage,
    SingleCameraData,
    MultiCameraData,
)
from maniunicon.utils.timestamp_accumulator import (
    get_accumulate_timestamp_idxs,
    TimestampAlignedBuffer,
)


class DummyRealSenseSensor(BaseSensor):
    """Dummy RealSense sensor that generates synthetic camera data without hardware communication."""

    def __init__(
        self,
        shared_storage: SharedStorage,
        camera_config: Dict[str, Dict],
        frequency: float = 30.0,  # 30 FPS
        verbose: bool = False,
        image_path: Optional[str] = None,
        default_depth: float = 1.0,
        warn_on_late: bool = True,
    ):
        """
        Initialize the dummy RealSense sensor.

        Args:
            shared_storage: Shared storage manager instance
            camera_config: Dictionary mapping camera names to their configurations
                Each config should have:
                - serial_number: str - The dummy camera's identifier (not used)
                - resolution: Tuple[int, int] - (width, height) of the RGB stream
                - fps: int - Frames per second to generate
                - transform: np.ndarray - Camera transform matrix
            frequency: Frequency to update the robot state in shared memory (in Hz)
            verbose: Whether to print debug information
            image_path: Optional path to image file to use instead of white image
            default_depth: Default depth value for all pixels
            warn_on_late: Whether to warn if the sensor is late
        """
        super().__init__(shared_storage, frequency)
        self.camera_config = camera_config
        self.verbose = verbose
        self.image_path = image_path
        self.default_depth = default_depth
        self.warn_on_late = warn_on_late
        self.record_buffer = None

        # Pre-generate dummy data for each camera
        self.camera_data = {}
        self._setup_dummy_data()

    def _setup_dummy_data(self):
        """Pre-generate dummy data for all cameras."""
        for cam_name in self.camera_config["camera_names"].keys():
            resolution = self.camera_config[cam_name]["resolution"]
            width, height = resolution

            # Generate color image
            if self.image_path is not None:
                try:
                    # Load image from path
                    color = cv2.imread(self.image_path)
                    if color is None:
                        raise ValueError(f"Could not load image from {self.image_path}")
                    # Convert BGR to RGB
                    color = color[..., ::-1]
                    # Resize to target resolution
                    color = cv2.resize(color, (width, height))
                except Exception as e:
                    if self.verbose:
                        print(
                            f"[DummyRealSenseSensor] Failed to load image {self.image_path}: {e}"
                        )
                        print(
                            f"[DummyRealSenseSensor] Falling back to white image for {cam_name}"
                        )
                    # Fall back to white image
                    color = np.ones((height, width, 3), dtype=np.uint8) * 255
            else:
                # Generate white image
                color = np.ones((height, width, 3), dtype=np.uint8) * 255

            # Generate constant depth
            depth = np.ones((height, width), dtype=np.float32) * self.default_depth

            # Generate dummy intrinsics (reasonable defaults for the resolution)
            fx = fy = max(width, height) * 0.8  # Approximate focal length
            cx, cy = width / 2, height / 2
            intr = np.array([fx, fy, cx, cy], dtype=np.float32)

            self.camera_data[cam_name] = {"color": color, "depth": depth, "intr": intr}

        if self.verbose:
            print(
                f"[DummyRealSenseSensor] Dummy data generated for {len(self.camera_data)} cameras"
            )

    @property
    def is_ready(self):
        """Dummy sensor is always ready once initialized."""
        return True

    def run(self):
        """Main loop that generates and writes dummy camera data."""
        try:
            if self.verbose:
                print("[DummyRealSenseSensor] Starting dummy sensor loop")

            # Initialize transforms and intrinsics
            transforms = []
            intrs = []
            camera_names = list(self.camera_config["camera_names"].keys())

            for cam_name in camera_names:
                transforms.append(self.camera_config[cam_name]["transform"])
                intrs.append(self.camera_data[cam_name]["intr"])

            transforms = np.stack(transforms)
            intrs = np.stack(intrs)

            # Setup rate limiting
            rate = RateLimiter(
                frequency=self.frequency,
                warn=self.warn_on_late,
                name="dummy_realsense_sensor",
            )

            # Timestamp tracking for each camera
            put_start_time = time.time()
            put_idxs = {cam_name: None for cam_name in camera_names}

            iter_idx = 0
            t_start = time.time()

            while self.shared_storage.is_running.value:
                try:
                    current_time = time.time()

                    # Generate data for each camera
                    camera_data = {}
                    for cam_name in camera_names:
                        fps = self.camera_config[cam_name]["fps"]

                        # Get timestamp indices for this camera
                        local_idxs, global_idxs, put_idxs[cam_name] = (
                            get_accumulate_timestamp_idxs(
                                timestamps=[current_time],
                                start_time=put_start_time,
                                dt=1.0 / fps,
                                next_global_idx=put_idxs[cam_name],
                                allow_negative=True,
                            )
                        )

                        # Create SingleCameraData for each step
                        for step_idx in global_idxs:
                            single_data = SingleCameraData(
                                color=self.camera_data[cam_name]["color"].copy(),
                                depth=self.camera_data[cam_name]["depth"].copy(),
                                intr=self.camera_data[cam_name]["intr"].copy(),
                                camera_receive_timestamp=current_time,
                                camera_capture_timestamp=current_time,
                                step_idx=step_idx,
                                timestamp=current_time,
                            )

                            # Write to shared storage
                            self.shared_storage.write_single_camera(
                                cam_name, single_data
                            )
                            camera_data[cam_name] = single_data

                    # Create fused multi-camera data if we have data from all cameras
                    if len(camera_data) == len(camera_names):
                        multi_camera_data = self._fuse_multi_camera(
                            camera_data, transforms, intrs
                        )

                        # Handle recording
                        if self.shared_storage.is_recording.value:
                            if self.record_buffer is None:
                                self.record_buffer = TimestampAlignedBuffer(
                                    self.shared_storage.record_start_time.value,
                                    self.shared_storage.record_dt.value,
                                    self.shared_storage.max_record_steps,
                                    overwrite=False,
                                )
                            self.record_buffer.add(
                                multi_camera_data.model_dump(),
                                timestamp=multi_camera_data.timestamp.item(),
                            )
                        else:
                            if self.record_buffer is not None:
                                self._dump_data()
                                self.record_buffer = None

                        # Write fused data to shared storage
                        self.shared_storage.write_multi_camera(multi_camera_data)

                    # Performance logging
                    iter_idx += 1
                    if self.verbose and iter_idx % 30 == 0:  # Log every 30 iterations
                        t_end = time.time()
                        duration = t_end - t_start
                        frequency = 30 / duration if duration > 0 else self.frequency
                        print(f"[DummyRealSenseSensor] Update rate: {frequency:.1f} Hz")
                        t_start = t_end

                except Exception as e:
                    traceback.print_exc()
                    print(f"Error in dummy RealSense sensor update loop: {e}")
                    self.shared_storage.error_state.value = True
                    break

                rate.sleep()

        except Exception as e:
            traceback.print_exc()
            print(f"Fatal error in dummy RealSense sensor: {e}")
            self.shared_storage.error_state.value = True
        finally:
            if self.verbose:
                print("[DummyRealSenseSensor] Sensor loop exiting")

    def _fuse_multi_camera(
        self,
        camera_data: Dict[str, SingleCameraData],
        transforms: np.ndarray,
        intrs: np.ndarray,
    ):
        """Fuse multiple dummy camera data into MultiCameraData."""
        depth_list = []
        color_list = []
        timestamps = []

        for camera_name, data in camera_data.items():
            color_list.append(data.color)
            depth_list.append(data.depth)
            timestamps.append(data.timestamp)

        colors = np.stack(color_list)
        depths = np.stack(depth_list)

        multi_camera_data = MultiCameraData(
            depths=depths,
            colors=colors,
            intrs=intrs,
            transforms=transforms,
            timestamp=np.array(np.mean(timestamps)),
        )
        return multi_camera_data

    def _dump_data(self):
        """Dump recorded data to file."""
        record_dir = self.shared_storage.get_record_dir()
        if not record_dir:
            print("Record directory not set, skipping data dump")
            return
        self.record_buffer.dump(name="dummy_realsense", dir=record_dir)

    def stop(self):
        """Stop the dummy sensor."""
        if self.verbose:
            print("Stopping dummy RealSense sensor...")
        super().stop()
