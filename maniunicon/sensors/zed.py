import torch.multiprocessing as mp
import time
import traceback
from typing import Dict, Tuple, Optional

import numpy as np
import pyzed.sl as sl
from threadpoolctl import threadpool_limits
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


class SingleZedCamera(mp.Process):
    """Individual camera process for capturing RGB and depth images from a single ZED camera."""

    def __init__(
        self,
        shared_storage: SharedStorage,
        camera_name: str,
        serial_number: Optional[int] = None,
        resolution: str = "HD720",  # HD720, HD1080, HD2K, VGA
        fps: int = 30,
        depth_mode: str = "ULTRA",  # PERFORMANCE, QUALITY, ULTRA, NEURAL
        coordinate_units: str = "METER",
        minimum_distance: float = 0.2,
        maximum_distance: float = 10.0,
        verbose: bool = False,
    ):
        super().__init__()
        self.shared_storage = shared_storage
        self.camera_name = camera_name
        self.serial_number = serial_number
        self.resolution = resolution
        self.fps = fps
        self.depth_mode = depth_mode
        self.coordinate_units = coordinate_units
        self.minimum_distance = minimum_distance
        self.maximum_distance = maximum_distance
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

    def _get_resolution_enum(self):
        """Convert resolution string to ZED enum."""
        resolution_map = {
            "HD2K": sl.RESOLUTION.HD2K,
            "HD1080": sl.RESOLUTION.HD1080,
            "HD720": sl.RESOLUTION.HD720,
            "VGA": sl.RESOLUTION.VGA,
        }
        return resolution_map.get(self.resolution, sl.RESOLUTION.HD720)

    def _get_depth_mode_enum(self):
        """Convert depth mode string to ZED enum."""
        depth_mode_map = {
            "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE,
            "QUALITY": sl.DEPTH_MODE.QUALITY,
            "ULTRA": sl.DEPTH_MODE.ULTRA,
            "NEURAL": sl.DEPTH_MODE.NEURAL,
        }
        return depth_mode_map.get(self.depth_mode, sl.DEPTH_MODE.ULTRA)

    def _get_coordinate_units_enum(self):
        """Convert coordinate units string to ZED enum."""
        units_map = {
            "MILLIMETER": sl.UNIT.MILLIMETER,
            "CENTIMETER": sl.UNIT.CENTIMETER,
            "METER": sl.UNIT.METER,
            "INCH": sl.UNIT.INCH,
            "FOOT": sl.UNIT.FOOT,
        }
        return units_map.get(self.coordinate_units, sl.UNIT.METER)

    def run(self):
        """Main camera capture loop."""

        try:
            # Limit threads for this process
            threadpool_limits(1)
            cv2.setNumThreads(1)

            # Create ZED camera object
            zed = sl.Camera()

            # Set initialization parameters
            init_params = sl.InitParameters()
            init_params.camera_resolution = self._get_resolution_enum()
            init_params.camera_fps = self.fps
            init_params.depth_mode = self._get_depth_mode_enum()
            init_params.coordinate_units = self._get_coordinate_units_enum()
            init_params.depth_minimum_distance = self.minimum_distance
            init_params.depth_maximum_distance = self.maximum_distance

            # Set serial number if provided
            if self.serial_number is not None:
                init_params.set_from_serial_number(self.serial_number)

            # Enable positional tracking if needed
            init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP

            # Open the camera
            try:
                err = zed.open(init_params)
                if err != sl.ERROR_CODE.SUCCESS:
                    print(f"Error opening ZED camera {self.camera_name}: {err}")
                    self.shared_storage.error_state.value = True
                    raise RuntimeError(f"Failed to open ZED camera: {err}")

                # Get camera calibration parameters
                camera_info = zed.get_camera_information()
                calibration_params = (
                    camera_info.camera_configuration.calibration_parameters
                )
                left_cam = calibration_params.left_cam

                # Extract intrinsics (fx, fy, cx, cy)
                intr = np.array([left_cam.fx, left_cam.fy, left_cam.cx, left_cam.cy])

                # Set runtime parameters
                runtime_params = sl.RuntimeParameters()
                runtime_params.sensing_mode = sl.SENSING_MODE.STANDARD
                runtime_params.confidence_threshold = 100
                runtime_params.texture_confidence_threshold = 100

                # Create Mat objects for retrieving data
                image = sl.Mat()
                depth = sl.Mat()

                # put frequency regulation
                put_idx = None
                put_start_time = self.put_start_time
                if put_start_time is None:
                    put_start_time = time.time()

                iter_idx = 0
                t_start = time.time()

                # Warm up the camera
                for _ in range(10):
                    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                        pass

                if self.verbose:
                    print(f"[SingleZedCamera {self.camera_name}] Camera initialized")

            except Exception as e:
                print(f"Error setting up camera {self.camera_name}: {e}")
                self.shared_storage.error_state.value = True
                raise e

            # Main loop
            try:
                while (
                    not self.stop_event.is_set()
                    and self.shared_storage.is_running.value
                ):
                    # Grab an image
                    if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                        receive_time = time.time()

                        # Retrieve left image (RGB)
                        zed.retrieve_image(image, sl.VIEW.LEFT)
                        # Convert from BGRA to RGB
                        color_bgra = image.get_data()
                        color = color_bgra[:, :, :3][:, :, ::-1]  # BGRA to RGB

                        # Retrieve depth map
                        zed.retrieve_measure(depth, sl.MEASURE.DEPTH)
                        depth_data = depth.get_data()

                        # Get timestamp from camera
                        timestamp = zed.get_timestamp(sl.TIME_REFERENCE.IMAGE)
                        # Convert from nanoseconds to seconds
                        capture_time = timestamp.get_nanoseconds() / 1e9

                        local_idxs, global_idxs, put_idx = (
                            get_accumulate_timestamp_idxs(
                                timestamps=[receive_time],
                                start_time=put_start_time,
                                dt=1.0 / self.fps,
                                next_global_idx=put_idx,
                                allow_negative=True,
                            )
                        )

                        for step_idx in global_idxs:
                            data = SingleCameraData(
                                color=color,
                                depth=depth_data,
                                intr=intr,
                                camera_receive_timestamp=receive_time,
                                camera_capture_timestamp=capture_time,
                                step_idx=step_idx,
                                timestamp=receive_time,
                            )
                            self.shared_storage.write_single_camera(
                                self.camera_name, data
                            )

                        # Signal ready
                        if iter_idx == 0:
                            self.ready_event.set()

                        t_end = time.time()
                        duration = t_end - t_start
                        frequency = np.round(1 / duration, 1)
                        t_start = t_end

                        iter_idx += 1
                        # Performance logging
                        if self.verbose and iter_idx % 30 == 0:  # Log every 30 frames
                            print(
                                f"[SingleZedCamera {self.camera_name}] FPS: {frequency:.1f}"
                            )

            except Exception as e:
                print(f"Error in camera {self.camera_name} capture loop: {e}")
                traceback.print_exc()
                self.shared_storage.error_state.value = True

        except Exception as e:
            print(f"Fatal error in camera {self.camera_name}: {e}")
            self.shared_storage.error_state.value = True
        finally:
            # Close the camera
            if "zed" in locals():
                zed.close()
            self.ready_event.set()  # Ensure ready event is set even on error
            if self.verbose:
                print(f"[SingleZedCamera {self.camera_name}] Process exiting")


class ZedSensor(BaseSensor):
    """Process that manages multiple ZED camera processes and produces stacked images and fused point clouds."""

    def __init__(
        self,
        shared_storage: SharedStorage,
        camera_config: Dict[str, Dict],
        frequency: float = 30.0,  # 30 FPS
        verbose: bool = False,
        warn_on_late: bool = True,
    ):
        """
        Initialize the ZED sensor.

        Args:
            shared_storage: Shared memory manager instance
            camera_config: Dictionary mapping camera names to their configurations
                Each config should have:
                - serial_number: Optional[int] - The ZED camera's serial number (None for auto-detect)
                - resolution: str - Resolution setting (HD720, HD1080, HD2K, VGA)
                - fps: int - Frames per second to capture
                - depth_mode: str - Depth mode (PERFORMANCE, QUALITY, ULTRA, NEURAL)
                - coordinate_units: str - Units for coordinates (METER, MILLIMETER, etc.)
                - minimum_distance: float - Minimum depth distance
                - maximum_distance: float - Maximum depth distance
            frequency: Frequency to update the robot state in shared memory (in Hz)
            verbose: Whether to print debug information
            warn_on_late: Whether to warn when the update loop is running late
        """

        super().__init__(shared_storage, frequency)
        self.camera_config = camera_config
        self.verbose = verbose
        self.warn_on_late = warn_on_late
        self.camera_processes: Dict[str, SingleZedCamera] = {}
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
                self.camera_windows[cam_name] = f"ZED Camera {cam_name}"

                # Load overlay image for this specific camera
                overlay_path = self.camera_config[cam_name].get("overlay_image", "none")
                if overlay_path and overlay_path != "none":
                    try:
                        overlay_img = cv2.imread(overlay_path)
                        if overlay_img is not None:
                            # Get resolution dimensions
                            resolution_str = self.camera_config[cam_name].get(
                                "resolution", "HD720"
                            )
                            resolution_map = {
                                "HD2K": (2208, 1242),
                                "HD1080": (1920, 1080),
                                "HD720": (1280, 720),
                                "VGA": (672, 376),
                            }
                            width, height = resolution_map.get(
                                resolution_str, (1280, 720)
                            )
                            overlay_img = cv2.resize(overlay_img, (width, height))
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
            camera_process = SingleZedCamera(
                shared_storage=self.shared_storage,
                camera_name=cam_name,
                serial_number=serial_number if serial_number != "auto" else None,
                resolution=self.camera_config[cam_name].get("resolution", "HD720"),
                fps=self.camera_config[cam_name].get("fps", 30),
                depth_mode=self.camera_config[cam_name].get("depth_mode", "ULTRA"),
                coordinate_units=self.camera_config[cam_name].get(
                    "coordinate_units", "METER"
                ),
                minimum_distance=self.camera_config[cam_name].get(
                    "minimum_distance", 0.2
                ),
                maximum_distance=self.camera_config[cam_name].get(
                    "maximum_distance", 10.0
                ),
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
                name="zed_sensor",
            )
            while self.shared_storage.is_running.value and not self.stop_event.is_set():
                try:
                    camera_data = {}
                    for camera_name in self.camera_processes.keys():
                        data = self.shared_storage.read_single_camera(camera_name)
                        if data is None:
                            print(
                                "[ZedSensor] No data received from camera:",
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
                    print(f"Error in ZED sensor update loop: {e}")
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

        self.record_buffer.dump(name="zed", dir=record_dir)

    def stop(self):
        """Stop the sensor and all camera processes."""

        if self.verbose:
            print("Stopping ZED sensor...")
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
