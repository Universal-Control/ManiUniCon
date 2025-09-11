import ctypes
from multiprocessing import Value, Event
from multiprocessing.managers import SharedMemoryManager
from typing import Any, Dict, List, Optional
from enum import Enum
from scipy.spatial.transform import Rotation as R
from omegaconf import OmegaConf

import numpy as np
from pydantic import BaseModel

from maniunicon.utils.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)
from maniunicon.utils.shared_memory.shared_memory_queue import SharedMemoryQueue, Empty
from maniunicon.utils.shared_memory.shared_memory_util import ArraySpec


class ControlMode(Enum):
    """Enum for robot control modes."""

    JOINT_POSITION = 0
    CARTESIAN = 1

    @classmethod
    def from_str(cls, mode_str: str) -> "ControlMode":
        """Convert string to ControlMode enum."""
        mode_map = {
            "joint": cls.JOINT_POSITION,
            "cartesian": cls.CARTESIAN,
        }
        if mode_str not in mode_map:
            raise ValueError(
                f"Invalid control mode: {mode_str}. Valid modes are: {list(mode_map.keys())}"
            )
        return mode_map[mode_str]

    def to_str(self) -> str:
        """Convert ControlMode enum to string."""
        mode_map = {
            self.JOINT_POSITION: "joint",
            self.CARTESIAN: "cartesian",
        }
        return mode_map[self]

    @classmethod
    def from_int(cls, mode_int: int) -> "ControlMode":
        """Convert integer to ControlMode enum."""
        try:
            return cls(mode_int)
        except ValueError:
            raise ValueError(
                f"Invalid control mode integer: {mode_int}. Valid values are: {[mode.value for mode in cls]}"
            )

    def to_int(self) -> int:
        """Convert ControlMode enum to integer."""
        return self.value


class RobotState(BaseModel):
    """Represents the current state of a robot arm.

    Performance Considerations:
    This class uses Pydantic's BaseModel for data validation and type safety. While this adds a small overhead
    compared to direct dictionary access from shared memory, the performance impact is typically negligible
    in robot control scenarios for the following reasons:

    1. Data validation and conversion overhead is in the microsecond range
    2. Typical control frequencies (tens to hundreds of Hz) are well within acceptable limits
    3. The overhead is small compared to other system latencies (network, sensor sampling, etc.)

    The benefits of using Pydantic models include:
    - Type safety and runtime validation
    - Better code maintainability and readability
    - Automatic data validation and conversion
    - Clear interface definition

    If extreme performance optimization is required (e.g., microsecond-level control), consider using
    direct dictionary access in the critical path. However, for most applications, the benefits of
    using Pydantic models outweigh the minimal performance cost.
    """

    model_config = {"arbitrary_types_allowed": True}

    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray
    tcp_position: np.ndarray  # Tool Center Point position [x, y, z]
    tcp_orientation: np.ndarray  # Quaternion [x, y, z, w]
    gripper_state: np.ndarray
    timestamp: (
        np.ndarray
    )  # state is multi-step, so we need an ndarray to store the timestamp of each step


class RobotAction(BaseModel):
    """Represents a control action for the robot."""

    model_config = {"arbitrary_types_allowed": True}

    joint_positions: Optional[np.ndarray] = None
    joint_velocities: Optional[np.ndarray] = None
    joint_torques: Optional[np.ndarray] = None
    tcp_position: Optional[np.ndarray] = None
    tcp_orientation: Optional[np.ndarray] = None
    gripper_state: Optional[np.ndarray] = (
        None  # TODO(zbzhu): change ndarray to a single float number
    )
    control_mode: str  # Will be validated against ControlMode enum
    timestamp: float
    target_timestamp: Optional[float] = None

    def __init__(self, **data):
        super().__init__(**data)
        # Validate control mode on initialization
        ControlMode.from_str(self.control_mode)


class SingleCameraData(BaseModel):
    """Represents the data from a single camera."""

    model_config = {"arbitrary_types_allowed": True}

    color: np.ndarray
    depth: np.ndarray
    camera_capture_timestamp: float
    camera_receive_timestamp: float
    timestamp: float
    step_idx: int
    intr: np.ndarray


class MultiCameraData(BaseModel):
    """Represents the data from multiple cameras."""

    model_config = {"arbitrary_types_allowed": True}

    depths: Optional[np.ndarray] = None
    colors: Optional[np.ndarray] = None
    intrs: Optional[np.ndarray] = None
    transforms: Optional[np.ndarray] = None
    timestamp: np.ndarray


class SharedStorage:
    """All-in-one shared storage between processes for robot control."""

    def __init__(
        self,
        shm_manager: SharedMemoryManager,
        robot_state_config: Dict[str, Any] | None = None,
        robot_action_config: Dict[str, Any] | None = None,
        camera_config: Dict[str, Any] | None = None,
        max_record_steps: int = 500,
    ):
        self.robot_state_config = robot_state_config
        self.robot_action_config = robot_action_config
        self.camera_config = camera_config
        self.max_record_steps = max_record_steps

        if self.camera_config is not None:
            camera_names = self.camera_config["camera_names"]
            for camera_name in camera_names:
                cam_config = self.camera_config[camera_name]
                camera_transform = np.eye(4)
                camera_transform[:3, 3] = cam_config["pose"]["position"]
                camera_transform[:3, :3] = R.from_quat(
                    cam_config["pose"]["orientation"], scalar_first=True
                ).as_matrix()
                cam_config.update(
                    {
                        "transform": camera_transform.tolist(),
                        "transform_T": np.ascontiguousarray(
                            camera_transform[:3, :3].T
                        ).tolist(),
                    }
                )

        # Create shared ring buffers
        self._create_shared_ring_buffers(shm_manager)
        self._create_shared_queues(shm_manager)

        # Control flags
        self.is_running = Value(ctypes.c_bool, True)
        self.is_recording = Value(ctypes.c_bool, False)
        self.record_start_time = Value(ctypes.c_double, 0.0)
        self.record_dt = Value(ctypes.c_double, 0.01)
        self.error_state = Value(ctypes.c_bool, False)

        # Synchronization events for synchronized policy-robot execution
        # Two-phase handshake protocol:
        # 1. Robot completes actions and sets robot_ready
        # 2. Policy waits for robot_ready, clears it, and starts inference
        # 3. Policy completes inference, sends actions, and sets policy_ready
        # 4. Robot waits for policy_ready, clears it, and executes actions
        # 5. Repeat from step 1
        self.robot_ready = Event()
        self.robot_ready.set()  # Initially set to True - robot ready to receive first actions

        self.policy_ready = Event()
        # Initially False - policy hasn't generated actions yet, so don't set it

        # Shared record directory (using a fixed-size char array)
        self.record_dir = Value(ctypes.c_char * 512)  # 512 bytes for path

    def _create_shared_queues(self, shm_manager: SharedMemoryManager):
        """Create shared queues for the robot action."""

        if self.robot_action_config is not None:
            robot_action_specs = [
                ArraySpec(
                    name="joint_positions",
                    shape=(self.robot_action_config["num_joints"],),
                    dtype=np.float64,
                ),
                ArraySpec(
                    name="joint_velocities",
                    shape=(self.robot_action_config["num_joints"],),
                    dtype=np.float64,
                ),
                ArraySpec(
                    name="joint_torques",
                    shape=(self.robot_action_config["num_joints"],),
                    dtype=np.float64,
                ),
                ArraySpec(name="tcp_position", shape=(3,), dtype=np.float64),
                ArraySpec(name="tcp_orientation", shape=(4,), dtype=np.float64),
                ArraySpec(name="gripper_state", shape=(1,), dtype=np.float64),
                ArraySpec(name="control_mode", shape=(1,), dtype=np.uint8),
                ArraySpec(name="timestamp", shape=(1,), dtype=np.float64),
                ArraySpec(name="target_timestamp", shape=(1,), dtype=np.float64),
            ]
            self.action_queue = SharedMemoryQueue(
                shm_manager,
                robot_action_specs,
                buffer_size=self.robot_action_config["buffer_size"],
            )

    def _create_shared_ring_buffers(self, shm_manager: SharedMemoryManager):
        """Create shared ring buffers for the robot state and image data."""

        if self.robot_state_config is not None:
            robot_state_specs = [
                ArraySpec(
                    name="joint_positions",
                    shape=(self.robot_state_config["num_joints"],),
                    dtype=np.float64,
                ),
                ArraySpec(
                    name="joint_velocities",
                    shape=(self.robot_state_config["num_joints"],),
                    dtype=np.float64,
                ),
                ArraySpec(
                    name="joint_torques",
                    shape=(self.robot_state_config["num_joints"],),
                    dtype=np.float64,
                ),
                ArraySpec(name="tcp_position", shape=(3,), dtype=np.float64),
                ArraySpec(name="tcp_orientation", shape=(4,), dtype=np.float64),
                ArraySpec(name="gripper_state", shape=(1,), dtype=np.float64),
                ArraySpec(name="timestamp", shape=(1,), dtype=np.float64),
            ]
            self.state_buffer = SharedMemoryRingBuffer(
                shm_manager,
                robot_state_specs,
                get_max_k=self.robot_state_config["get_max_k"],
                get_time_budget=self.robot_state_config["get_time_budget"],
                put_desired_frequency=self.robot_state_config["put_desired_frequency"],
            )

        if self.camera_config is not None:
            self.camera_buffers = {}
            for cam_name in self.camera_config["camera_names"]:
                cam_config = self.camera_config[cam_name]
                resolution = cam_config["resolution"]
                shape = resolution[::-1]
                camera_specs = [
                    ArraySpec(name="color", shape=tuple(shape) + (3,), dtype=np.uint8),
                    ArraySpec(name="depth", shape=shape, dtype=np.float32),
                    ArraySpec(
                        name="camera_capture_timestamp", shape=(1,), dtype=np.float64
                    ),
                    ArraySpec(
                        name="camera_receive_timestamp", shape=(1,), dtype=np.float64
                    ),
                    ArraySpec(name="timestamp", shape=(1,), dtype=np.float64),
                    ArraySpec(name="step_idx", shape=(1,), dtype=np.int32),
                    ArraySpec(name="intr", shape=(4,), dtype=np.float32),
                ]
                self.camera_buffers[cam_name] = SharedMemoryRingBuffer(
                    shm_manager,
                    camera_specs,
                    get_max_k=cam_config["single_camera_get_max_k"],
                    get_time_budget=cam_config["single_camera_get_time_budget"],
                    put_desired_frequency=cam_config[
                        "single_camera_put_desired_frequency"
                    ],
                )

            multi_camera_specs = [
                ArraySpec(
                    name="depths",
                    shape=(
                        len(self.camera_config["camera_names"]),
                        resolution[1],
                        resolution[0],
                    ),
                    dtype=np.float32,
                ),
                ArraySpec(
                    name="colors",
                    shape=(
                        len(self.camera_config["camera_names"]),
                        resolution[1],
                        resolution[0],
                        3,
                    ),
                    dtype=np.uint8,
                ),
                ArraySpec(
                    name="intrs",
                    shape=(len(self.camera_config["camera_names"]), 4),
                    dtype=np.float32,
                ),
                ArraySpec(
                    name="transforms",
                    shape=(len(self.camera_config["camera_names"]), 4, 4),
                    dtype=np.float32,
                ),
                ArraySpec(name="timestamp", shape=(1,), dtype=np.float64),
            ]
            self.multi_camera_buffer = SharedMemoryRingBuffer(
                shm_manager,
                multi_camera_specs,
                get_max_k=self.camera_config["multi_camera_get_max_k"],
                get_time_budget=self.camera_config["multi_camera_get_time_budget"],
                put_desired_frequency=self.camera_config[
                    "multi_camera_put_desired_frequency"
                ],
            )

    def write_state(self, state: RobotState):
        """Write a new state to the ring buffer."""

        if self.state_buffer is None:
            raise ValueError(
                "State buffer not created. Please initialize the state buffer by passing a robot_state_config to the SharedStorage constructor."
            )
        self.state_buffer.put(
            {
                "joint_positions": state.joint_positions,
                "joint_velocities": state.joint_velocities,
                "joint_torques": state.joint_torques,
                "tcp_position": state.tcp_position,
                "tcp_orientation": state.tcp_orientation,
                "gripper_state": state.gripper_state,
                "timestamp": state.timestamp,
            }
        )

    def read_state(self, k: int | None = None) -> RobotState | None:
        """Read the latest state from the ring buffer."""

        if self.state_buffer is None:
            raise ValueError(
                "State buffer not created. Please initialize the state buffer by passing a robot_state_config to the SharedStorage constructor."
            )

        if k is None:
            state = self.state_buffer.get()
        else:
            state = self.state_buffer.get_last_k(k)
        if state is not None:
            return RobotState(
                joint_positions=state["joint_positions"],
                joint_velocities=state["joint_velocities"],
                joint_torques=state["joint_torques"],
                tcp_position=state["tcp_position"],
                tcp_orientation=state["tcp_orientation"],
                gripper_state=state["gripper_state"],
                timestamp=state["timestamp"],
            )
        else:
            return None

    def write_action(self, action: RobotAction):
        """Write a new action to the shared memory."""

        if self.action_queue is None:
            raise ValueError(
                "Action buffer not created. Please initialize the action buffer by passing a robot_action_config to the SharedStorage constructor."
            )
        control_mode = ControlMode.from_str(action.control_mode).to_int()
        self.action_queue.put(
            {
                "joint_positions": action.joint_positions,
                "joint_velocities": action.joint_velocities,
                "joint_torques": action.joint_torques,
                "tcp_position": action.tcp_position,
                "tcp_orientation": action.tcp_orientation,
                "gripper_state": action.gripper_state,
                "control_mode": control_mode,
                "timestamp": action.timestamp,
                "target_timestamp": action.target_timestamp,
            }
        )

    def read_all_action(self) -> List[RobotAction]:
        """Read all actions from the shared memory."""

        if self.action_queue is None:
            raise ValueError(
                "Action buffer not created. Please initialize the action buffer by passing a robot_action_config to the SharedStorage constructor."
            )
        actions = []
        try:
            outputs = self.action_queue.get_all()
        except Empty:
            return actions
        n_actions = len(outputs["timestamp"])
        for i in range(n_actions):
            control_mode = ControlMode.from_int(outputs["control_mode"][i]).to_str()
            actions.append(
                RobotAction(
                    joint_positions=outputs["joint_positions"][i],
                    joint_velocities=outputs["joint_velocities"][i],
                    joint_torques=outputs["joint_torques"][i],
                    tcp_position=outputs["tcp_position"][i],
                    tcp_orientation=outputs["tcp_orientation"][i],
                    gripper_state=outputs["gripper_state"][i],
                    control_mode=control_mode,
                    timestamp=outputs["timestamp"][i],
                    target_timestamp=outputs["target_timestamp"][i],
                )
            )
        return actions

    def read_action(self) -> RobotAction:
        """Read the current action from the shared memory."""

        if self.action_queue is None:
            raise ValueError(
                "Action buffer not created. Please initialize the action buffer by passing a robot_action_config to the SharedStorage constructor."
            )
        action = self.action_queue.get()
        control_mode = ControlMode.from_int(action["control_mode"]).to_str()
        return RobotAction(
            joint_positions=action["joint_positions"],
            joint_velocities=action["joint_velocities"],
            joint_torques=action["joint_torques"],
            tcp_position=action["tcp_position"],
            tcp_orientation=action["tcp_orientation"],
            gripper_state=action["gripper_state"],
            control_mode=control_mode,
            timestamp=action["timestamp"],
            target_timestamp=action["target_timestamp"],
        )

    def write_single_camera(self, cam_name: str, cam_data: SingleCameraData):
        """Write a new single camera data to the shared memory."""

        if self.camera_buffers[cam_name] is None:
            raise ValueError(
                f"Camera buffer for {cam_name} not created. Please initialize the camera buffer by passing a camera_config to the SharedStorage constructor."
            )
        self.camera_buffers[cam_name].put(
            {
                "color": cam_data.color,
                "depth": cam_data.depth,
                "camera_capture_timestamp": cam_data.camera_capture_timestamp,
                "camera_receive_timestamp": cam_data.camera_receive_timestamp,
                "timestamp": cam_data.timestamp,
                "step_idx": cam_data.step_idx,
                "intr": cam_data.intr,
            }
        )

    def read_single_camera(
        self,
        cam_name: str,
        k: int | None = None,
    ) -> SingleCameraData:
        """Read the latest single camera data from the shared memory."""

        if self.camera_buffers[cam_name] is None:
            raise ValueError(
                f"Camera buffer for {cam_name} not created. Please initialize the camera buffer by passing a camera_config to the SharedStorage constructor."
            )
        if k is None:
            data = self.camera_buffers[cam_name].get()
        else:
            data = self.camera_buffers[cam_name].get_last_k(k)
        self.tmp_out_single = data
        if data is not None:
            if data["color"] is None:
                print(
                    f"\033[93mWarning: Color data is None for camera {cam_name}\033[0m"
                )
            if data["depth"] is None:
                print(
                    f"\033[93mWarning: Depth data is None for camera {cam_name}\033[0m"
                )
            return SingleCameraData(
                color=data["color"],
                depth=data["depth"],
                camera_capture_timestamp=data["camera_capture_timestamp"],
                camera_receive_timestamp=data["camera_receive_timestamp"],
                timestamp=data["timestamp"],
                step_idx=data["step_idx"],
                intr=data["intr"],
            )
        else:
            return None

    def write_multi_camera(self, cam_data: MultiCameraData):
        """Write a new multi camera data to the shared memory."""

        if self.multi_camera_buffer is None:
            raise ValueError(
                "Multi camera buffer not created. Please initialize the multi camera buffer by passing a camera_config to the SharedStorage constructor."
            )
        self.multi_camera_buffer.put(
            {
                "depths": cam_data.depths,
                "colors": cam_data.colors,
                "intrs": cam_data.intrs,
                "transforms": cam_data.transforms,
                "timestamp": cam_data.timestamp,
            }
        )

    def read_multi_camera(self, k: int | None = None) -> MultiCameraData:
        """Read the latest multi camera data from the shared memory."""

        if self.multi_camera_buffer is None:
            raise ValueError(
                "Multi camera buffer not created. Please initialize the multi camera buffer by passing a camera_config to the SharedStorage constructor."
            )
        if k is None:
            data = self.multi_camera_buffer.get()
        else:
            data = self.multi_camera_buffer.get_last_k(k)
        if data is not None:
            return MultiCameraData(
                depths=data["depths"],
                colors=data["colors"],
                intrs=data["intrs"],
                transforms=data["transforms"],
                timestamp=data["timestamp"],
            )
        else:
            return None

    def set_record_dir(self, record_dir: str | None):
        """Set the current recording directory for all processes."""
        if record_dir is None:
            record_dir = ""
        # Encode string to bytes and ensure it fits in the buffer
        record_dir_bytes = record_dir.encode("utf-8")[
            :511
        ]  # Leave space for null terminator
        with self.record_dir.get_lock():
            self.record_dir.value = record_dir_bytes

    def clear_record_dir(self):
        """Clear the current recording directory."""
        with self.record_dir.get_lock():
            self.record_dir.value = b""

    def get_record_dir(self) -> str:
        """Get the current recording directory."""
        with self.record_dir.get_lock():
            record_dir_bytes = self.record_dir.value
        # Decode bytes to string
        return record_dir_bytes.decode("utf-8")

    def start_record(self, start_time: float, dt: float):
        """Start recording."""
        self.record_start_time.value = start_time
        self.record_dt.value = dt
        self.is_recording.value = True

    def stop_record(self):
        """Stop recording."""
        self.is_recording.value = False
