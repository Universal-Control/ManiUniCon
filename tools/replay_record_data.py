import os
import time
import argparse
import traceback
import signal
import sys
import numpy as np
import cv2
import torch
import matplotlib
import matplotlib.cm as cm

from maniunicon.utils.replay_buffer import ReplayBuffer
from maniunicon.robot_interface.meshcat import MeshcatInterface


# Global flag for clean shutdown
shutdown_requested = False


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, torch.Tensor):
        depth = depth_map.detach().clone().squeeze().numpy()
    elif isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        if isinstance(depth_map, torch.Tensor):
            valid_mask = valid_mask.detach().numpy()
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    if isinstance(depth_map, torch.Tensor):
        img_colored = torch.from_numpy(img_colored_np).float()
    elif isinstance(depth_map, np.ndarray):
        img_colored = img_colored_np

    return img_colored


def create_colorbar_image(height, width=50):
    """Create a colorbar image from red to blue (warm to cool)."""
    # Create gradient from 0 to 1
    gradient = np.linspace(0, 1, height)[:, np.newaxis]
    gradient = np.tile(gradient, (1, width))

    # Apply red-to-blue colormap (reversed coolwarm)
    colorbar = cm.coolwarm_r(gradient)
    colorbar_rgb = (colorbar[:, :, :3] * 255).astype(np.uint8)

    return colorbar_rgb


def signal_handler(sig, frame):
    """Handle Ctrl+C and other signals for clean shutdown."""
    global shutdown_requested
    print("\nShutdown requested. Cleaning up...")
    shutdown_requested = True
    cv2.destroyAllWindows()
    sys.exit(0)


def setup_robot_visualizer(robot_config=None):
    """Setup meshcat robot visualizer using pink meshcat."""
    if robot_config is None:
        # Default XArm6 configuration based on meshcat_xarm6.yaml
        robot_config = {
            "urdf_path": "assets/xarm6_urdf/xarm6_robot_white.urdf",
            "urdf_package_dirs": ["assets/xarm6_urdf"],
            "num_joints": 6,
            "frequency": 200.0,
            "joint_limits": {
                "min": [-6.28, -2.059, -3.927, -6.28, -1.69297, -6.28],
                "max": [6.28, 2.0944, 0.19198, 6.28, 3.14, 6.28],
            },
            "velocity_limits": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
            "torque_limits": [150.0, 150.0, 150.0, 150.0, 150.0, 150.0],
            "tcp_frame": "link6",
            "init_qpos": [0.014, -0.64, -0.39, 0.0, 1.0, 0.0],
        }

    # Create meshcat interface for visualization
    meshcat_interface = MeshcatInterface(robot_config)

    if meshcat_interface.connect():
        print(f"Robot visualizer available at: {meshcat_interface.viewer.url()}")
        return meshcat_interface
    else:
        print(
            "Failed to connect to robot visualizer, falling back to simple visualization"
        )
        return None


def visualize_robot_state(
    meshcat_interface,
    joint_positions,
    tcp_position=None,
    tcp_orientation=None,
    check_consistency=True,
):
    """Visualize robot state using pink meshcat robot."""
    if meshcat_interface is not None and meshcat_interface.is_connected():
        try:
            # Update robot configuration with joint positions
            if joint_positions is not None:
                # Create action to update robot state
                from maniunicon.utils.shared_memory.shared_storage import RobotAction

                action = RobotAction(
                    control_mode="joint",
                    joint_positions=joint_positions,
                    gripper_state=np.array([0.0]),  # Default gripper state
                    timestamp=time.time(),
                )

                # Send action to update visualization
                meshcat_interface.send_action(action)

                # Check consistency between joint positions and TCP pose if both are available
                if (
                    check_consistency
                    and tcp_position is not None
                    and tcp_orientation is not None
                ):
                    try:
                        # Compute forward kinematics from joint positions
                        fk_position, fk_orientation = (
                            meshcat_interface.forward_kinematics(joint_positions)
                        )

                        # Calculate position error (Euclidean distance)
                        pos_error = np.linalg.norm(
                            np.array(tcp_position) - np.array(fk_position)
                        )

                        # Calculate orientation error (angle between quaternions)
                        # Normalize quaternions
                        tcp_quat = np.array(tcp_orientation) / np.linalg.norm(
                            tcp_orientation
                        )
                        fk_quat = np.array(fk_orientation) / np.linalg.norm(
                            fk_orientation
                        )

                        # Compute angle between quaternions (dot product gives cos of half angle)
                        quat_dot = np.clip(np.abs(np.dot(tcp_quat, fk_quat)), 0, 1)
                        orientation_error = 2 * np.arccos(
                            quat_dot
                        )  # Convert to radians

                        # Define tolerance thresholds
                        pos_tolerance = 0.01  # 1cm
                        orientation_tolerance = 0.1  # ~5.7 degrees

                        if (
                            pos_error > pos_tolerance
                            or orientation_error > orientation_tolerance
                        ):
                            print("⚠️  TCP pose inconsistency detected:")
                            print(
                                f"   Position error: {pos_error:.4f} m "
                                f"(tolerance: {pos_tolerance} m)"
                            )
                            print(
                                f"   Orientation error: {orientation_error:.4f} rad "
                                f"(tolerance: {orientation_tolerance} rad)"
                            )
                            print(f"   Data TCP position: {tcp_position}")
                            print(f"   FK TCP position: {fk_position}")
                            print(f"   Data TCP orientation: {tcp_orientation}")
                            print(f"   FK TCP orientation: {fk_orientation}")
                        else:
                            print(
                                f"✅ TCP pose consistency check passed "
                                f"(pos error: {pos_error:.4f}m, "
                                f"orient error: {orientation_error:.4f}rad)"
                            )

                    except Exception as e:
                        print(f"Error during FK consistency check: {e}")

        except Exception as e:
            print(f"Error updating robot visualization: {e}")
    else:
        # Fallback to simple visualization if meshcat interface is not available
        print("Using fallback visualization (no robot model available)")


def add_text_to_image(
    image, text, position=(10, 30), font_scale=1.0, color=(255, 255, 255)
):
    """Add text to an image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2

    # Add black background for better text visibility
    cv2.putText(image, text, position, font, font_scale, (0, 0, 0), thickness + 1)
    # Add white text
    cv2.putText(image, text, position, font, font_scale, color, thickness)

    return image


def replay_data(
    zarr_path,
    frequency=10.0,
    start_episode=0,
    robot_config=None,
    check_consistency_freq=10,
    save_video_path=None,
):
    """Replay the recorded zarr data."""
    global shutdown_requested

    # Load replay buffer
    print(f"Loading replay buffer from {zarr_path}")
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

    # Setup robot visualizer
    meshcat_interface = setup_robot_visualizer(robot_config)

    # Get data info
    print(f"Replay buffer contains {replay_buffer.n_episodes} episodes")

    # Initialize video writers if save_video_path is provided
    video_writers = {}
    if save_video_path is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_video_path), exist_ok=True)
        print(f"Video will be saved to: {save_video_path}")

    # Process each episode starting from start_episode
    for episode_idx in range(start_episode, replay_buffer.n_episodes):
        if shutdown_requested:
            break

        print(f"\nReplaying episode {episode_idx}")

        # Get episode slice indices
        start_idx = 0
        if episode_idx > 0:
            start_idx = int(replay_buffer.episode_ends[episode_idx - 1])
        end_idx = int(replay_buffer.episode_ends[episode_idx])

        n_steps = end_idx - start_idx
        print(f"Episode has {n_steps} steps (indices {start_idx} to {end_idx-1})")

        # Check available data
        obs = replay_buffer.data["obs"]
        has_images = "images" in obs
        has_depths = "depths" in obs
        has_model_depths = "model_depths" in obs
        has_joint_positions = "joint_positions" in obs
        has_tcp_pose = "tcp_position" in obs and "tcp_orientation" in obs

        print(
            f"Available data - Images: {has_images}, Depths: {has_depths}, Model Depths: {has_model_depths}, "
            f"Joint positions: {has_joint_positions}, TCP pose: {has_tcp_pose}"
        )

        if has_images:
            # Get camera names
            camera_names = list(obs["images"].keys())
            print(f"Found cameras: {camera_names}")

            # Create OpenCV windows
            for cam_name in camera_names:
                cv2.namedWindow(f"Camera {cam_name}", cv2.WINDOW_AUTOSIZE)

        # Replay the episode
        step_delay = 1.0 / frequency

        for local_step in range(n_steps):
            if shutdown_requested:
                break

            start_time = time.time()

            # Calculate global step index
            global_step = start_idx + local_step

            # Display images
            if has_images:
                for cam_name in camera_names:
                    # Get RGB image
                    img = np.array(obs["images"][cam_name][global_step])
                    if img.dtype == np.float32 or img.dtype == np.float64:
                        img = (img * 255).astype(np.uint8)

                    # Initialize images list with RGB
                    images_to_display = [img]
                    image_labels = ["RGB"]

                    # Get depth image if available
                    depth_img = None
                    if has_depths:
                        depth = np.array(obs["depths"][cam_name][global_step])
                        depth_img = colorize_depth_maps(
                            depth[:, :, 0], min_depth=0.0, max_depth=1.5
                        )[0].transpose(
                            1, 2, 0
                        )  # Convert to HWC format
                        if depth_img is not None:
                            depth_img = (depth_img * 255).astype(np.uint8)
                            images_to_display.append(depth_img)
                            image_labels.append("Depth")

                    # Get model depth image if available
                    model_depth_img = None
                    if has_model_depths:
                        model_depth = np.array(
                            obs["model_depths"][cam_name][global_step]
                        )
                        model_depth_img = colorize_depth_maps(
                            model_depth[:, :, 0], min_depth=0.0, max_depth=1.5
                        )[0].transpose(
                            1, 2, 0
                        )  # Convert to HWC format
                        if model_depth_img is not None:
                            model_depth_img = (model_depth_img * 255).astype(np.uint8)
                            images_to_display.append(model_depth_img)
                            image_labels.append("Model Depth")

                    # Concatenate images horizontally
                    if len(images_to_display) > 1:
                        # Ensure all images have the same height
                        target_height = img.shape[0]
                        resized_images = []

                        for i, img_to_resize in enumerate(images_to_display):
                            if img_to_resize.shape[0] != target_height:
                                # Resize to match height while maintaining aspect ratio
                                aspect_ratio = (
                                    img_to_resize.shape[1] / img_to_resize.shape[0]
                                )
                                new_width = int(target_height * aspect_ratio)
                                img_to_resize = cv2.resize(
                                    img_to_resize, (new_width, target_height)
                                )

                            # Add label to image
                            if i < len(image_labels):
                                img_to_resize = add_text_to_image(
                                    img_to_resize, image_labels[i], position=(10, 30)
                                )

                            resized_images.append(img_to_resize)

                        # Concatenate horizontally
                        combined_img = np.hstack(resized_images)
                    else:
                        combined_img = img

                    # Convert RGB to BGR for OpenCV display
                    combined_img_bgr = cv2.cvtColor(combined_img, cv2.COLOR_RGB2BGR)
                    cv2.imshow(f"Camera {cam_name}", combined_img_bgr)

                    # Initialize video writer on first frame if needed
                    if save_video_path is not None and cam_name not in video_writers:
                        # Create video filename for this camera and episode
                        base_name = os.path.splitext(save_video_path)[0]
                        ext = os.path.splitext(save_video_path)[1]
                        if ext == "":
                            ext = ".mp4"
                        video_filename = f"{base_name}_ep{episode_idx}_{cam_name}{ext}"

                        # Get video dimensions from current frame
                        height, width = combined_img_bgr.shape[:2]

                        # Create video writer
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            video_filename, fourcc, frequency, (width, height)
                        )

                        if not video_writer.isOpened():
                            print(
                                f"Warning: Could not open video writer for {video_filename}"
                            )
                            video_writer = None
                        else:
                            print(
                                f"Video writer initialized for {video_filename} ({width}x{height})"
                            )

                        video_writers[cam_name] = video_writer

                    # Write frame to video if video writer exists
                    if (
                        cam_name in video_writers
                        and video_writers[cam_name] is not None
                    ):
                        video_writers[cam_name].write(combined_img_bgr)

            # Update robot visualization
            joint_positions = None
            tcp_position = None
            tcp_orientation = None

            if has_joint_positions:
                joint_positions = np.array(obs["joint_positions"][global_step])

            if has_tcp_pose:
                tcp_position = np.array(obs["tcp_position"][global_step])
                tcp_orientation = np.array(obs["tcp_orientation"][global_step])

            # Only check consistency every N steps to avoid spam
            should_check_consistency = (
                check_consistency_freq > 0 and local_step % check_consistency_freq == 0
            )

            visualize_robot_state(
                meshcat_interface,
                joint_positions,
                tcp_position,
                tcp_orientation,
                should_check_consistency,
            )

            # Print step info
            if local_step % 10 == 0:
                print(f"Step {local_step}/{n_steps-1} (global: {global_step})")
                if has_joint_positions:
                    print(f"  Joint positions: {joint_positions}")
                if has_tcp_pose:
                    print(f"  TCP position: {tcp_position}")
                    print(f"  TCP orientation: {tcp_orientation}")

                # Print depth statistics if available
                if has_images and has_depths:
                    for cam_name in camera_names:
                        depth = np.array(obs["depths"][cam_name][global_step])
                        valid_depth = depth[~np.isnan(depth)]
                        if len(valid_depth) > 0:
                            print(
                                f"  {cam_name} Depth - Min: {np.min(valid_depth):.3f}m, "
                                f"Max: {np.max(valid_depth):.3f}m, "
                                f"Mean: {np.mean(valid_depth):.3f}m"
                            )

                if has_images and has_model_depths:
                    for cam_name in camera_names:
                        model_depth = np.array(
                            obs["model_depths"][cam_name][global_step]
                        )
                        valid_model_depth = model_depth[~np.isnan(model_depth)]
                        if len(valid_model_depth) > 0:
                            print(
                                f"  {cam_name} Model Depth - Min: "
                                f"{np.min(valid_model_depth):.3f}m, "
                                f"Max: {np.max(valid_model_depth):.3f}m, "
                                f"Mean: {np.mean(valid_model_depth):.3f}m"
                            )

            # Handle keyboard input with shorter timeout
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit requested")
                shutdown_requested = True
                break
            elif key == ord(" "):
                print("Paused - press any key to continue")
                cv2.waitKey(0)
            elif key == ord("n"):
                print("Next episode")
                break
            elif key == ord("r"):
                print("Restarting episode")
                break

            # Control playback speed
            elapsed_time = time.time() - start_time
            sleep_time = step_delay - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

        # Close video writers for this episode
        if save_video_path is not None:
            for cam_name, video_writer in video_writers.items():
                if video_writer is not None:
                    video_writer.release()
                    print(f"Video saved for camera {cam_name}")
            video_writers.clear()

        if shutdown_requested:
            break

        print(f"Finished episode {episode_idx}")

        # Ask if user wants to continue to next episode
        if episode_idx < replay_buffer.n_episodes - 1:
            print(
                "Press 'c' to continue to next episode, 'q' to quit, or any other key to replay this episode"
            )
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q"):
                shutdown_requested = True
                break
            elif key == ord("c"):
                continue
            else:
                episode_idx -= 1  # Replay current episode

    # Cleanup
    cv2.destroyAllWindows()
    if meshcat_interface is not None:
        meshcat_interface.disconnect()
    print("Replay completed")


def main():
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    parser = argparse.ArgumentParser(
        description="Replay recorded zarr data with visualization"
    )
    parser.add_argument("zarr_path", type=str, help="Path to the zarr file to replay")
    parser.add_argument(
        "--frequency",
        type=float,
        default=10.0,
        help="Playback frequency (default: 10.0 Hz)",
    )
    parser.add_argument(
        "--start_episode",
        type=int,
        default=0,
        help="Episode to start replay from (default: 0)",
    )
    parser.add_argument(
        "--robot_name",
        type=str,
        default="xarm6",
        help="Robot name for visualization (default: xarm6)",
    )
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="assets/xarm6_urdf/xarm6_robot_white.urdf",
        help="Path to URDF file (default: assets/xarm6_urdf/xarm6_robot_white.urdf)",
    )
    parser.add_argument(
        "--urdf_package_dirs",
        type=str,
        nargs="+",
        default=["assets/xarm6_urdf"],
        help="URDF package directories (default: assets/xarm6_urdf)",
    )
    parser.add_argument(
        "--num_joints",
        type=int,
        default=6,
        help="Number of robot joints (default: 6)",
    )
    parser.add_argument(
        "--tcp_frame",
        type=str,
        default="link6",
        help="TCP frame name (default: link6)",
    )
    parser.add_argument(
        "--check_consistency_freq",
        type=int,
        default=10,
        help="Frequency of TCP pose consistency checks (default: 10, 0 to disable)",
    )
    parser.add_argument(
        "--save_video_path",
        type=str,
        default=None,
        help="Path to save the video (default: None)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file not found: {args.zarr_path}")
        return

    # Create robot configuration
    robot_config = {
        "urdf_path": args.urdf_path,
        "urdf_package_dirs": args.urdf_package_dirs,
        "num_joints": args.num_joints,
        "frequency": 200.0,
        "joint_limits": {
            "min": [-6.28, -2.059, -3.927, -6.28, -1.69297, -6.28],
            "max": [6.28, 2.0944, 0.19198, 6.28, 3.14, 6.28],
        },
        "velocity_limits": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        "torque_limits": [150.0, 150.0, 150.0, 150.0, 150.0, 150.0],
        "tcp_frame": args.tcp_frame,
        "init_qpos": [0.014, -0.64, -0.39, 0.0, 1.0, 0.0],
    }

    print(f"Starting replay of {args.zarr_path}")
    print(f"Using XArm6 robot with URDF: {args.urdf_path}")
    print("Controls:")
    print("  'q' - Quit")
    print("  'space' - Pause/unpause")
    print("  'n' - Next episode")
    print("  'r' - Restart current episode")
    print("  'c' - Continue to next episode (when prompted)")
    print("  Ctrl+C - Force quit")

    try:
        replay_data(
            args.zarr_path,
            args.frequency,
            args.start_episode,
            robot_config,
            args.check_consistency_freq,
            args.save_video_path,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during replay: {e}")
        print(traceback.print_exc())
    finally:
        cv2.destroyAllWindows()
        print("Cleanup completed")


if __name__ == "__main__":
    main()
