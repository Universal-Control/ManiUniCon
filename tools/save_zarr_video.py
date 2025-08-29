import os
import argparse
import traceback
import numpy as np
import cv2
import matplotlib
from tqdm import tqdm

from maniunicon.utils.replay_buffer import ReplayBuffer


def colorize_depth_maps(
    depth_map, min_depth, max_depth, cmap="Spectral", valid_mask=None
):
    """
    Colorize depth maps.
    """
    assert len(depth_map.shape) >= 2, "Invalid dimension"

    if isinstance(depth_map, np.ndarray):
        depth = depth_map.copy().squeeze()
    # reshape to [ (B,) H, W ]
    if depth.ndim < 3:
        depth = depth[np.newaxis, :, :]

    # colorize
    cm_func = matplotlib.colormaps[cmap]
    depth = ((depth - min_depth) / (max_depth - min_depth)).clip(0, 1)
    img_colored_np = cm_func(depth, bytes=False)[:, :, :, 0:3]  # value from 0 to 1
    img_colored_np = np.rollaxis(img_colored_np, 3, 1)

    if valid_mask is not None:
        valid_mask = valid_mask.squeeze()  # [H, W] or [B, H, W]
        if valid_mask.ndim < 3:
            valid_mask = valid_mask[np.newaxis, np.newaxis, :, :]
        else:
            valid_mask = valid_mask[:, np.newaxis, :, :]
        valid_mask = np.repeat(valid_mask, 3, axis=1)
        img_colored_np[~valid_mask] = 0

    return img_colored_np


def save_zarr_videos(
    zarr_path,
    output_path,
    frequency=30.0,
    min_depth=0.0,
    max_depth=1.5,
    video_codec="mp4v",
):
    """Save videos from zarr data without visualization."""

    # Load replay buffer
    print(f"Loading replay buffer from {zarr_path}")
    replay_buffer = ReplayBuffer.create_from_path(zarr_path, mode="r")

    # Get data info
    print(f"Replay buffer contains {replay_buffer.n_episodes} episodes")

    # Prepare output directory
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(output_path)[0]
    ext = os.path.splitext(output_path)[1]
    if ext == "":
        ext = ".mp4"

    # Process each episode
    for episode_idx in tqdm(
        range(replay_buffer.n_episodes), desc="Processing episodes"
    ):
        print(f"\nProcessing episode {episode_idx}")

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

        print(
            f"Available data - Images: {has_images}, Depths: {has_depths}, Model Depths: {has_model_depths}"
        )

        if not has_images:
            print("No image data found, skipping episode")
            continue

        # Get camera names
        camera_names = list(obs["images"].keys())
        print(f"Found cameras: {camera_names}")

        # Initialize video writers for each camera and data type
        video_writers = {}

        # Process each step in the episode
        for local_step in tqdm(
            range(n_steps), desc=f"Episode {episode_idx}", leave=False
        ):
            global_step = start_idx + local_step

            # Process each camera
            for cam_name in camera_names:
                # Process RGB
                try:
                    rgb_img = np.array(obs["images"][cam_name][global_step])
                    if rgb_img.dtype == np.float32 or rgb_img.dtype == np.float64:
                        rgb_img = (rgb_img * 255).astype(np.uint8)

                    # Convert RGB to BGR for OpenCV
                    rgb_img_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)

                    # Initialize RGB video writer if not exists
                    rgb_key = f"{cam_name}_rgb"
                    if rgb_key not in video_writers:
                        rgb_filename = (
                            f"{base_name}_ep{episode_idx:03d}_{cam_name}_rgb{ext}"
                        )
                        height, width = rgb_img_bgr.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*video_codec)
                        writer = cv2.VideoWriter(
                            rgb_filename, fourcc, frequency, (width, height)
                        )

                        if writer.isOpened():
                            video_writers[rgb_key] = writer
                            print(f"  Created RGB video writer: {rgb_filename}")
                        else:
                            print(
                                f"  Warning: Could not create RGB video writer for {rgb_filename}"
                            )
                            video_writers[rgb_key] = None

                    # Write RGB frame
                    if video_writers[rgb_key] is not None:
                        video_writers[rgb_key].write(rgb_img_bgr)

                except Exception as e:
                    print(
                        f"  Error processing RGB for {cam_name} at step {local_step}: {e}"
                    )

                # Process Depth
                if has_depths:
                    try:
                        depth = np.array(obs["depths"][cam_name][global_step])
                        depth_img = colorize_depth_maps(
                            depth[:, :, 0], min_depth=min_depth, max_depth=max_depth
                        )[0].transpose(
                            1, 2, 0
                        )  # Convert to HWC format
                        depth_img = (depth_img * 255).astype(np.uint8)

                        # Convert RGB to BGR for OpenCV
                        depth_img_bgr = cv2.cvtColor(depth_img, cv2.COLOR_RGB2BGR)

                        # Initialize depth video writer if not exists
                        depth_key = f"{cam_name}_depth"
                        if depth_key not in video_writers:
                            depth_filename = (
                                f"{base_name}_ep{episode_idx:03d}_{cam_name}_depth{ext}"
                            )
                            height, width = depth_img_bgr.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*video_codec)
                            writer = cv2.VideoWriter(
                                depth_filename, fourcc, frequency, (width, height)
                            )

                            if writer.isOpened():
                                video_writers[depth_key] = writer
                                print(f"  Created depth video writer: {depth_filename}")
                            else:
                                print(
                                    f"  Warning: Could not create depth video writer for {depth_filename}"
                                )
                                video_writers[depth_key] = None

                        # Write depth frame
                        if video_writers[depth_key] is not None:
                            video_writers[depth_key].write(depth_img_bgr)

                    except Exception as e:
                        print(
                            f"  Error processing depth for {cam_name} at step {local_step}: {e}"
                        )

                # Process Model Depth
                if has_model_depths:
                    try:
                        model_depth = np.array(
                            obs["model_depths"][cam_name][global_step]
                        )
                        model_depth_img = colorize_depth_maps(
                            model_depth[:, :, 0],
                            min_depth=min_depth,
                            max_depth=max_depth,
                        )[0].transpose(
                            1, 2, 0
                        )  # Convert to HWC format
                        model_depth_img = (model_depth_img * 255).astype(np.uint8)

                        # Convert RGB to BGR for OpenCV
                        model_depth_img_bgr = cv2.cvtColor(
                            model_depth_img, cv2.COLOR_RGB2BGR
                        )

                        # Initialize model depth video writer if not exists
                        model_depth_key = f"{cam_name}_model_depth"
                        if model_depth_key not in video_writers:
                            model_depth_filename = f"{base_name}_ep{episode_idx:03d}_{cam_name}_model_depth{ext}"
                            height, width = model_depth_img_bgr.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*video_codec)
                            writer = cv2.VideoWriter(
                                model_depth_filename, fourcc, frequency, (width, height)
                            )

                            if writer.isOpened():
                                video_writers[model_depth_key] = writer
                                print(
                                    f"  Created model depth video writer: {model_depth_filename}"
                                )
                            else:
                                print(
                                    f"  Warning: Could not create model depth video writer for {model_depth_filename}"
                                )
                                video_writers[model_depth_key] = None

                        # Write model depth frame
                        if video_writers[model_depth_key] is not None:
                            video_writers[model_depth_key].write(model_depth_img_bgr)

                    except Exception as e:
                        print(
                            f"  Error processing model depth for {cam_name} at step {local_step}: {e}"
                        )

        # Close all video writers for this episode
        for key, writer in video_writers.items():
            if writer is not None:
                writer.release()
                print(f"  Saved video: {key}")
        video_writers.clear()

        print(f"Completed episode {episode_idx}")

    print("All videos saved successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="Save videos from zarr data without visualization"
    )
    parser.add_argument("zarr_path", type=str, help="Path to the zarr file")
    parser.add_argument(
        "output_path",
        type=str,
        help="Output path template (e.g., 'videos/recording.mp4')",
    )
    parser.add_argument(
        "--frequency",
        type=float,
        default=30.0,
        help="Video frame rate (default: 30.0 Hz)",
    )
    parser.add_argument(
        "--min_depth",
        type=float,
        default=0.0,
        help="Minimum depth for colorization (default: 0.0)",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=1.5,
        help="Maximum depth for colorization (default: 1.5)",
    )
    parser.add_argument(
        "--video_codec",
        type=str,
        default="mp4v",
        help="Video codec (default: mp4v)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.zarr_path):
        print(f"Error: Zarr file not found: {args.zarr_path}")
        return

    print(f"Input zarr file: {args.zarr_path}")
    print(f"Output template: {args.output_path}")
    print(f"Video settings: {args.frequency} Hz, codec: {args.video_codec}")
    print(f"Depth range: {args.min_depth}m - {args.max_depth}m")
    print()

    try:
        save_zarr_videos(
            args.zarr_path,
            args.output_path,
            args.frequency,
            args.min_depth,
            args.max_depth,
            args.video_codec,
        )
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error during video export: {e}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
