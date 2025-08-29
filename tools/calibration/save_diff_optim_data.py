import time
import hydra
import numpy as np
from loguru import logger as lgr
from omegaconf import OmegaConf
from multiprocessing.managers import SharedMemoryManager
import ast

from maniunicon.utils.shared_memory.shared_storage import SharedStorage


def load_joint_values_from_file(file_path):
    """
    Load joint values from a text file

    Args:
        file_path: Path to the text file containing joint values

    Returns:
        list: List of joint value arrays
    """
    joint_values_list = []
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                # Parse the line as a Python list
                joint_values = ast.literal_eval(line)
                joint_values_list.append(np.array(joint_values))

    lgr.info(
        f"Loaded {len(joint_values_list)} joint configurations " f"from {file_path}"
    )
    return joint_values_list


def capture_single_frame(robot_interface, shared_storage):
    """
    Capture a single RGB frame and current robot joint values

    Args:
        robot_interface: UR5 robot interface object
        shared_storage: Shared storage object for camera data

    Returns:
        tuple: (rgb_image, joint_values)
    """
    lgr.info("Capturing single frame...")

    # Get current robot state from robot interface
    state = robot_interface.get_state()
    joint_values = state.joint_positions
    lgr.info(f"Current joint values: {joint_values}")

    # Capture RGB image from camera sensor process
    time.sleep(3.0)  # Allow camera to stabilize
    camera_data = shared_storage.read_multi_camera()
    rgb_image = camera_data.colors[0]

    lgr.info(f"Captured RGB image with shape: {rgb_image.shape}")
    return rgb_image, joint_values


def run_multi_frame_calibration(cfg, joint_values_file_path):
    """
    Run the complete multi-frame camera calibration process

    Args:
        cfg: Hydra config
        joint_values_file_path: Path to the file containing joint values
    """

    lgr.info("Starting multi-frame camera calibration...")

    try:
        # Step 1: Initialize Robot Interface
        robot_interface = hydra.utils.instantiate(cfg.robot.robot_interface)
        robot_interface.reset_to_init()

        # Connect to robot
        if not robot_interface.connect():
            raise RuntimeError("Failed to connect to robot")

        lgr.info("Successfully connected to UR5 robot")

        # Step 2: Start Camera Sensor Process
        shm_manager = SharedMemoryManager()
        shm_manager.start()
        shared_storage = SharedStorage(
            shm_manager=shm_manager,
            robot_state_config=cfg.data.robot_state_config,
            robot_action_config=cfg.data.robot_action_config,
            camera_config=cfg.data.camera_config,
        )

        # Start camera sensors
        sensors = {
            name: hydra.utils.instantiate(
                sensor_cfg,
                shared_storage=shared_storage,
            )
            for name, sensor_cfg in cfg.sensors.items()
        }

        for sensor_name, sensor in sensors.items():
            sensor.start()
            print(f"{sensor_name} started.")

        # Step 3: Load joint values from file
        joint_configurations = load_joint_values_from_file(joint_values_file_path)

        # Step 4: Capture frames for each joint configuration
        rgb_images = []
        captured_joint_values = []

        for i, target_joints in enumerate(joint_configurations):
            lgr.info(f"Processing configuration {i+1}/{len(joint_configurations)}")
            lgr.info(f"Target joint values: {target_joints}")

            # Set robot to target joint configuration
            if not robot_interface.move_to_joint_positions(target_joints):
                raise RuntimeError(f"Failed to move to joint positions {target_joints}")

            # Wait for the robot to reach position and stabilize
            time.sleep(1.0)

            # Capture frame
            rgb_image, actual_joint_values = capture_single_frame(
                robot_interface, shared_storage
            )

            rgb_images.append(rgb_image)
            captured_joint_values.append(actual_joint_values)

            lgr.info(f"Captured frame {i+1} with shape: {rgb_image.shape}")

        # Step 5: Save all data to combined npz file
        rgb_images_array = np.array(rgb_images)
        captured_joint_values_array = np.array(captured_joint_values)
        target_joint_values_array = np.array(joint_configurations)

        # Check if captured joint values are close enough to target joint values
        joint_tolerance = 0.01  # radians
        joint_differences = np.abs(
            captured_joint_values_array - target_joint_values_array
        )
        max_joint_differences = np.max(joint_differences, axis=1)

        assert np.all(
            max_joint_differences <= joint_tolerance
        ), f"Joint values exceeded tolerance of {joint_tolerance} radians"
        lgr.info("All joint configurations are within tolerance")

        np.savez(
            "multi_frame_calibration_data.npz",
            rgb_images=rgb_images_array,
            captured_joint_values=captured_joint_values_array,
            target_joint_values=target_joint_values_array,
        )
        robot_interface.reset_to_init()

        lgr.info(
            f"Saved {len(rgb_images)} frames to " "multi_frame_calibration_data.npz"
        )
        lgr.info(f"RGB images shape: {rgb_images_array.shape}")
        lgr.info(f"Captured joint values shape: {captured_joint_values_array.shape}")
        lgr.info(f"Target joint values shape: {target_joint_values_array.shape}")

    except Exception as e:
        lgr.error(f"Calibration failed: {e}")
        raise
    finally:
        # Cleanup camera sensors
        for sensor in sensors.values():
            sensor.stop()
        shared_storage.is_running.value = False
        shm_manager.shutdown()

        # Cleanup robot interface
        if "robot_interface" in locals():
            robot_interface.disconnect()
            lgr.info("Robot disconnected")

        print("All systems stopped.")


@hydra.main(
    version_base=None,
    config_path="../configs",
    config_name="default",
)
def main(cfg):
    """
    Example usage of the multi-frame camera calibration functions
    """
    OmegaConf.register_new_resolver("eval", eval)

    # Path to the joint values file
    joint_values_file_path = "./assets/ur5_diff_optim_joints_l515.txt"

    # Run calibration
    run_multi_frame_calibration(cfg=cfg, joint_values_file_path=joint_values_file_path)


if __name__ == "__main__":
    main()
