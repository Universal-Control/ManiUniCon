from multiprocessing.managers import SharedMemoryManager
import numpy as np

from maniunicon.utils.shared_memory.shared_memory_ring_buffer import (
    SharedMemoryRingBuffer,
)
from maniunicon.utils.shared_memory.shared_memory_util import ArraySpec


def get_robot_state_buffer(
    shm_manager: SharedMemoryManager,
    num_joints: int = 7,
    get_time_budget: float = 0.01,
    get_max_k: int = 20,
    put_fps: float = 100,
):
    array_specs = [
        ArraySpec(name="joint_positions", shape=(num_joints,), dtype=np.float64),
        ArraySpec(name="joint_velocities", shape=(num_joints,), dtype=np.float64),
        ArraySpec(name="joint_torques", shape=(num_joints,), dtype=np.float64),
        ArraySpec(name="tcp_position", shape=(3,), dtype=np.float64),
        ArraySpec(name="tcp_orientation", shape=(4,), dtype=np.float64),
        ArraySpec(name="timestamp", shape=(1,), dtype=np.float64),
    ]
    return SharedMemoryRingBuffer(
        shm_manager,
        array_specs,
        get_max_k,
        get_time_budget,
        put_desired_frequency=put_fps,
    )


def get_robot_action_buffer(
    shm_manager: SharedMemoryManager,
    num_joints: int = 7,
    get_time_budget: float = 0.01,
    get_max_k: int = 10,
    put_fps: float = 200,
):
    array_specs = [
        ArraySpec(name="joint_positions", shape=(num_joints,), dtype=np.float64),
        ArraySpec(name="joint_velocities", shape=(num_joints,), dtype=np.float64),
        ArraySpec(name="joint_torques", shape=(num_joints,), dtype=np.float64),
        ArraySpec(name="tcp_position", shape=(3,), dtype=np.float64),
        ArraySpec(name="tcp_orientation", shape=(4,), dtype=np.float64),
        ArraySpec(name="timestamp", shape=(1,), dtype=np.float64),
        ArraySpec(name="control_mode", shape=(1,), dtype=str),
    ]
    return SharedMemoryRingBuffer(
        shm_manager,
        array_specs,
        get_max_k,
        get_time_budget,
        put_desired_frequency=put_fps,
    )


def get_single_amera_buffer(
    shm_manager: SharedMemoryManager,
    width: int,
    height: int,
    get_time_budget: float = 0.2,
    get_max_k: int = 10,
    put_fps: float = 60,
):
    array_specs = [
        ArraySpec(name="rgb", shape=(width, height, 3), dtype=np.uint8),
        ArraySpec(name="depth", shape=(width, height, 1), dtype=np.float32),
        ArraySpec(name="timestamp", shape=(1,), dtype=np.float32),
        ArraySpec(name="intr", shape=(4,), dtype=np.float32),
    ]
    return SharedMemoryRingBuffer(
        shm_manager,
        array_specs,
        get_max_k,
        get_time_budget,
        put_desired_frequency=put_fps,
    )


def get_multi_camera_buffer(
    shm_manager: SharedMemoryManager,
    num_cameras: int,
    width: int,
    height: int,
    npoints: int,
    get_time_budget: float = 0.3,
    get_max_k: int = 10,
    put_fps: float = 60,
):
    array_specs = [
        ArraySpec(name="pcds", shape=(num_cameras, npoints, 6), dtype=np.float32),
        ArraySpec(name="rgbs", shape=(num_cameras, width, height, 3), dtype=np.uint8),
        ArraySpec(
            name="depths", shape=(num_cameras, width, height, 1), dtype=np.float32
        ),
        ArraySpec(name="intrs", shape=(num_cameras, 4), dtype=np.float32),
        ArraySpec(name="transforms", shape=(num_cameras, 4, 4), dtype=np.float32),
    ]
    return SharedMemoryRingBuffer(
        shm_manager,
        array_specs,
        get_max_k,
        get_time_budget,
        put_desired_frequency=put_fps,
    )
