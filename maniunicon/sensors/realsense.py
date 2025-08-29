import torch.multiprocessing as mp
import time
import traceback
from typing import Dict, Tuple

import numpy as np
import pyrealsense2 as rs
from threadpoolctl import threadpool_limits
import cv2
from numba import njit
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


@njit
def filter_vectors(v, rgb):
    norms = np.sqrt((v**2).sum(axis=1))
    valid = norms > 0
    points = v[valid]
    colors = rgb[valid]
    return points, colors


@njit
def transform_points(points, transform, transform_T):
    points = points @ transform_T
    points += transform
    return points


def get_color_from_tex_coords(tex_coords, color_image):
    us = (tex_coords[:, 0] * 640).astype(int)
    vs = (tex_coords[:, 1] * 480).astype(int)

    us = np.clip(us, 0, 639)
    vs = np.clip(vs, 0, 479)

    colors = color_image[vs, us]

    return colors


class SingleRealsenseCamera(mp.Process):
    """Individual camera process for capturing RGB images from a single RealSense camera."""

    def __init__(
        self,
        shared_storage: SharedStorage,
        camera_name: str,
        serial_number: str,
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30,
        depth_hole_filling: bool = False,
        verbose: bool = False,
    ):
        super().__init__()
        self.shared_storage = shared_storage
        self.camera_name = camera_name
        self.serial_number = serial_number
        self.resolution = resolution
        self.fps = fps
        self.depth_hole_filling = depth_hole_filling
        self.verbose = verbose
        self.put_start_time = None

        # Process control
        self.stop_event = mp.Event()
        self.ready_event = mp.Event()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()

    @property
    def is_ready(self):
        return self.ready_event.is_set()

    def start(self, wait=True, put_start_time=None):
        self.put_start_time = put_start_time
        super().start()
        if wait:
            self.start_wait()

    def stop(self, wait=True):
        self.stop_event.set()
        if wait:
            self.join()

    def start_wait(self):
        self.ready_event.wait()

    def end_wait(self):
        self.join()

    def run(self):
        """Main camera capture loop."""

        try:
            # Limit threads for this process
            threadpool_limits(1)
            cv2.setNumThreads(1)

            # configure realsense
            rs_config = rs.config()

            # enable streams
            width, height = self.resolution
            rs_config.enable_stream(
                rs.stream.color, width, height, rs.format.bgr8, self.fps
            )
            rs_config.enable_stream(
                rs.stream.depth, width, height, rs.format.z16, self.fps
            )

            # create aligner
            align_to = rs.stream.color
            aligner = rs.align(align_to)

            # optionally set depth hole filling
            if self.depth_hole_filling:
                hole_filling = rs.hole_filling_filter(mode=2)

            # start pipeline
            try:
                rs_config.enable_device(self.serial_number)

                # pipeline
                pipeline = rs.pipeline()
                pipeline_profile = pipeline.start(rs_config)
                depth_scale = (
                    pipeline_profile.get_device().first_depth_sensor().get_depth_scale()
                )
                intr = (
                    pipeline_profile.get_stream(rs.stream.color)
                    .as_video_stream_profile()
                    .get_intrinsics()
                )
                intr = np.array([intr.fx, intr.fy, intr.ppx, intr.ppy])
                # report global time
                d = pipeline_profile.get_device().first_color_sensor()
                d.set_option(rs.option.global_time_enabled, 1)

                # put frequency regulation
                put_idx = None
                put_start_time = self.put_start_time
                if put_start_time is None:
                    put_start_time = time.time()

                iter_idx = 0
                t_start = time.time()

                # iterate through the first few frames to get the camera ready
                for _ in range(10):
                    pipeline.wait_for_frames()

                if self.verbose:
                    print(
                        f"[SingleRealsenseCamera {self.camera_name}] Camera initialized"
                    )

            except Exception as e:
                print(f"Error setting up camera {self.camera_name}: {e}")
                self.shared_storage.error_state.value = True
                raise e

            # main loop
            try:
                while (
                    not self.stop_event.is_set()
                    and self.shared_storage.is_running.value
                ):
                    # Wait for frames
                    frameset = pipeline.wait_for_frames()
                    frameset = aligner.process(frameset)
                    receive_time = time.time()

                    # Get color frame
                    color_frame = frameset.get_color_frame()
                    # Convert from BGR to RGB
                    color = np.asarray(color_frame.get_data())[..., ::-1]

                    # realsense report in ms
                    capture_time = color_frame.get_timestamp() / 1000.0

                    depth_frame = frameset.get_depth_frame()
                    if self.depth_hole_filling:
                        # Apply hole filling filter
                        depth_frame = hole_filling.process(depth_frame)

                    # working with frequency restriction first
                    depth = np.array(depth_frame.get_data()) * depth_scale

                    local_idxs, global_idxs, put_idx = get_accumulate_timestamp_idxs(
                        timestamps=[receive_time],
                        start_time=put_start_time,
                        dt=1.0 / self.fps,
                        next_global_idx=put_idx,
                        allow_negative=True,
                    )

                    for step_idx in global_idxs:
                        data = SingleCameraData(
                            color=color,
                            depth=depth,
                            intr=intr,
                            camera_receive_timestamp=receive_time,
                            camera_capture_timestamp=capture_time,
                            step_idx=step_idx,
                            timestamp=receive_time,
                        )
                        self.shared_storage.write_single_camera(self.camera_name, data)

                    # signal ready
                    if iter_idx == 0:
                        self.ready_event.set()

                    t_end = time.time()
                    duration = t_end - t_start
                    frequency = np.round(1 / duration, 1)
                    t_start = t_end

                    iter_idx += 1
                    # performance logging
                    if self.verbose and iter_idx % 30 == 0:  # Log every 30 frames
                        print(
                            f"[SingleRealsenseCamera {self.camera_name}] FPS: {frequency:.1f}"
                        )

            except Exception as e:
                print(f"Error in camera {self.camera_name} capture loop: {e}")
                import traceback

                traceback.print_exc()
                self.shared_storage.error_state.value = True

        except Exception as e:
            print(f"Fatal error in camera {self.camera_name}: {e}")
            self.shared_storage.error_state.value = True
        finally:
            rs_config.disable_all_streams()
            self.ready_event.set()  # Ensure ready event is set even on error
            if self.verbose:
                print(f"[SingleRealsenseCamera {self.camera_name}] Process exiting")


class RealSenseSensor(BaseSensor):
    """Process that manages multiple RealSense camera processes and produces stacked images and fused point clouds."""

    def __init__(
        self,
        shared_storage: SharedStorage,
        camera_config: Dict[str, Dict],
        frequency: float = 30.0,  # 30 FPS
        verbose: bool = False,
        warn_on_late: bool = True,
    ):
        """
        Initialize the RealSense sensor.

        Args:
            shared_memory: Shared memory manager instance
            camera_config: Dictionary mapping camera names to their configurations
                Each config should have:
                - serial_number: str - The RealSense camera's serial number
                - resolution: Tuple[int, int] - (width, height) of the RGB stream
                - fps: int - Frames per second to capture
            frequency: Frequency to update the robot state in shared memory (in Hz)
            verbose: Whether to print debug information
        """

        super().__init__(shared_storage, frequency)
        self.camera_config = camera_config
        self.verbose = verbose
        self.warn_on_late = warn_on_late
        self.camera_processes: Dict[str, SingleRealsenseCamera] = {}
        self.stop_event = mp.Event()
        self.record_buffer = None

        # Display configuration
        self.display_cameras = {}  # Track which cameras should be displayed
        self.camera_windows = {}  # Track window names for each camera
        self.camera_overlays = {}  # Track overlay images for each camera

        # Check which cameras have display enabled and setup their configurations
        for cam_name, serial_number in self.camera_config["camera_names"].items():
            if self.camera_config[cam_name].get("display", False):
                self.display_cameras[cam_name] = True
                self.camera_windows[cam_name] = f"RealSense Camera {cam_name}"

                # Load overlay image for this specific camera
                overlay_path = self.camera_config[cam_name].get("overlay_image", "none")
                if overlay_path and overlay_path != "none":
                    try:
                        overlay_img = cv2.imread(overlay_path)
                        if overlay_img is not None:
                            # Resize to camera resolution
                            resolution = self.camera_config[cam_name]["resolution"]
                            overlay_img = cv2.resize(
                                overlay_img, (resolution[0], resolution[1])
                            )
                            self.camera_overlays[cam_name] = overlay_img
                            if self.verbose:
                                print(
                                    f"Overlay image loaded for {cam_name} from {overlay_path}"
                                )
                        else:
                            if self.verbose:
                                print(
                                    f"Failed to load overlay image for {cam_name} from {overlay_path}"
                                )
                    except Exception as e:
                        if self.verbose:
                            print(f"Error loading overlay image for {cam_name}: {e}")

    def _setup_display(self):
        """Create display windows for cameras that have display enabled."""
        for cam_name in self.display_cameras.keys():
            window_name = self.camera_windows[cam_name]
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            if self.verbose:
                print(f"Display window '{window_name}' created for camera {cam_name}")

    def _setup_camera_processes(self):
        """Initialize and start all camera processes."""

        for cam_name, serial_number in self.camera_config["camera_names"].items():
            # Create camera process
            camera_process = SingleRealsenseCamera(
                shared_storage=self.shared_storage,
                camera_name=cam_name,
                serial_number=serial_number,
                resolution=self.camera_config[cam_name]["resolution"],
                fps=self.camera_config[cam_name]["fps"],
                verbose=self.verbose,
            )

            self.camera_processes[cam_name] = camera_process

            # Start the camera process
            camera_process.start(wait=False)

            # Small delay between camera starts to avoid conflicts
            time.sleep(0.5)

        # Wait for all cameras to be ready
        if self.verbose:
            print("Waiting for all cameras to be ready...")

        for cam_name, camera_process in self.camera_processes.items():
            camera_process.start_wait()
            if self.verbose:
                print(f"Camera {cam_name} is ready")

    def _cleanup_camera_processes(self):
        """Stop and cleanup all camera processes."""

        for cam_name, camera_process in self.camera_processes.items():
            try:
                camera_process.stop(wait=True)
                if self.verbose:
                    print(f"Camera process {cam_name} stopped")
            except Exception as e:
                print(f"Error stopping camera process {cam_name}: {e}")

    @property
    def is_ready(self):
        """Check if all camera processes are ready."""

        return all(camera.is_ready for camera in self.camera_processes.values())

    def run(self):
        """Loop that updates robot state with latest camera data."""

        threadpool_limits(16)
        cv2.setNumThreads(16)
        try:
            # Setup camera processes
            self._setup_camera_processes()

            # Create display windows for cameras that need them
            if self.display_cameras:
                self._setup_display()

            if self.verbose:
                print("All cameras ready, starting update loop")

            self.transforms = None
            self.intrs = None

            rate = RateLimiter(
                frequency=self.frequency,
                warn=self.warn_on_late,
                name="realsense_sensor",
            )
            while self.shared_storage.is_running.value and not self.stop_event.is_set():
                try:
                    camera_data = {}
                    for camera_name in self.camera_processes.keys():
                        data = self.shared_storage.read_single_camera(camera_name)
                        if data is None:
                            print(
                                "[RealSenseSensor] No data received from camera:",
                                camera_name,
                            )
                            rate.sleep()
                            continue
                        camera_data[camera_name] = data
                    multi_camera_data = self.fuse_multi_camera(camera_data)

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

                    # Update shared memory (this will trigger camera data to be included)
                    self.shared_storage.write_multi_camera(multi_camera_data)

                    # Display the multi-camera view
                    self._display_multi_camera(multi_camera_data)

                except Exception as e:
                    traceback.print_exc()
                    print(f"Error in RealSense sensor update loop: {e}")
                    self.shared_storage.error_state.value = True
                    break

                rate.sleep()

        finally:
            self._cleanup_camera_processes()

    def _dump_data(self):
        record_dir = self.shared_storage.get_record_dir()
        if not record_dir:
            print("Record directory not set, skipping data dump")
            return

        self.record_buffer.dump(name="realsense", dir=record_dir)

    def stop(self):
        """Stop the sensor and all camera processes."""

        if self.verbose:
            print("Stopping RealSense sensor...")
        self.stop_event.set()
        self._cleanup_camera_processes()
        for window_name in self.camera_windows.values():
            cv2.destroyWindow(window_name)
        super().stop()

    def fuse_multi_camera(self, camera_data: Dict[str, SingleCameraData]):
        depth_list = []
        color_list = []
        intr_list = []
        transforms = []

        timestamps = []
        for camera_name, data in camera_data.items():
            color = data.color
            depth = data.depth
            intr = data.intr
            color_list.append(color)
            depth_list.append(depth)
            timestamps.append(data.timestamp)
            if self.intrs is None:
                intr_list.append(intr)
            if self.transforms is None:
                transforms.append(self.camera_config[camera_name]["transform"])

        colors = np.stack(color_list)
        depths = np.stack(depth_list)

        if self.intrs is None:
            self.intrs = np.stack(intr_list)
        if self.transforms is None:
            self.transforms = np.stack(transforms)

        multi_camera_data = MultiCameraData(
            depths=depths,
            colors=colors,
            intrs=self.intrs,
            transforms=self.transforms,
            timestamp=np.array(np.mean(timestamps)),
        )
        return multi_camera_data

    def _display_multi_camera(self, multi_camera_data: "MultiCameraData"):
        """Display individual camera RGB images in separate windows.

        Args:
            multi_camera_data: MultiCameraData instance containing camera data
        """
        if not self.display_cameras:
            return

        try:
            camera_names = list(self.camera_config["camera_names"].keys())

            for cam_idx, cam_name in enumerate(camera_names):
                # Only display cameras that have display enabled
                if cam_name not in self.display_cameras:
                    continue

                # Get RGB image and convert from RGB to BGR for OpenCV
                rgb = cv2.cvtColor(multi_camera_data.colors[cam_idx], cv2.COLOR_RGB2BGR)

                # Apply overlay if provided for this specific camera
                if cam_name in self.camera_overlays:
                    overlay_img = self.camera_overlays[cam_name]
                    # Get alpha value from camera config, default to 0.3 if not specified
                    overlay_alpha = self.camera_config[cam_name].get(
                        "overlay_alpha", 0.3
                    )
                    # Blend images using alpha blending
                    rgb = cv2.addWeighted(
                        rgb, 1 - overlay_alpha, overlay_img, overlay_alpha, 0
                    )

                # Display the RGB image in the camera's window
                window_name = self.camera_windows[cam_name]
                cv2.imshow(window_name, rgb)

            cv2.waitKey(1)  # Non-blocking wait for key press

        except Exception as e:
            if self.verbose:
                print(f"Error in display: {e}")
