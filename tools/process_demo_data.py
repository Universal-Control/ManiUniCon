import os
import time
import argparse
import numpy as np
import glob
from pathlib import Path

from maniunicon.utils.misc import dict_apply
from maniunicon.utils.replay_buffer import ReplayBuffer


def find_episode_dirs(data_dir):
    """Find all episode directories in the data directory."""
    episode_pattern = os.path.join(data_dir, "episode_*")
    episodes = glob.glob(episode_pattern)
    # Natural sort by extracting the episode number
    return sorted(episodes, key=lambda x: int(Path(x).name.split("_")[1]))


def detect_control_mode(action_data):
    """Auto-detect control mode from action data."""
    if action_data is None:
        return None

    # Check which position fields have valid (non-NaN) data
    has_valid_joint = False
    has_valid_cartesian = False

    if "joint_positions" in action_data:
        joint_data = action_data["joint_positions"]
        if joint_data is not None and len(joint_data) > 0:
            # Check if data is not all NaN or zeros
            if not np.all(np.isnan(joint_data)) and not np.all(joint_data == 0):
                has_valid_joint = True

    if "tcp_position" in action_data and "tcp_orientation" in action_data:
        tcp_pos = action_data["tcp_position"]
        tcp_ori = action_data["tcp_orientation"]
        if (
            tcp_pos is not None
            and tcp_ori is not None
            and len(tcp_pos) > 0
            and len(tcp_ori) > 0
        ):
            # Check if data is not all NaN or zeros
            if not np.all(np.isnan(tcp_pos)) and not np.all(np.isnan(tcp_ori)):
                if not (np.all(tcp_pos == 0) and np.all(tcp_ori == 0)):
                    has_valid_cartesian = True

    # Check if control_mode field exists, but only use it if the data validates it
    if "control_mode" in action_data:
        mode = action_data["control_mode"]
        if isinstance(mode, np.ndarray) and len(mode) > 0:
            # Get the most common mode if it's an array
            unique_modes = np.unique(mode)
            if len(unique_modes) == 1:
                labeled_mode = str(unique_modes[0])
                # Validate the labeled mode against actual data
                if labeled_mode == "joint" and has_valid_joint:
                    return "joint"
                elif labeled_mode == "cartesian" and has_valid_cartesian:
                    return "cartesian"
                else:
                    print(
                        f"Warning: control_mode field says '{labeled_mode}' but data validation suggests otherwise"
                    )

    # Determine control mode based on available valid data
    if has_valid_joint and not has_valid_cartesian:
        return "joint"
    elif has_valid_cartesian and not has_valid_joint:
        return "cartesian"
    elif has_valid_joint and has_valid_cartesian:
        # If both are available, prefer cartesian as it was the original default
        print("Warning: Both joint and cartesian data found, defaulting to cartesian")
        return "cartesian"
    else:
        print("Warning: No valid position data found in action")
        return None


def load_episode_data(episode_dir):
    """Load state, action, and realsense data from an episode directory."""
    episode_dir = Path(episode_dir)
    episode_data = {}

    # Load state data
    state_path = episode_dir / "state.npz"
    if not state_path.exists():
        print(f"Warning: state.npz not found in {episode_dir}")
        state_data = None
    else:
        state_data = dict(np.load(state_path, allow_pickle=True))
        episode_data["state"] = state_data

    # Load action data
    action_path = episode_dir / "action.npz"
    if not action_path.exists():
        print(f"Warning: action.npz not found in {episode_dir}")
        action_data = None
    else:
        action_data = dict(np.load(action_path, allow_pickle=True))
        episode_data["action"] = action_data

    # Load realsense data
    realsense_path = episode_dir / "realsense.npz"
    if not realsense_path.exists():
        print(f"Warning: realsense.npz not found in {episode_dir}")
        realsense_data = None
    else:
        realsense_data = dict(np.load(realsense_path, allow_pickle=True))
        episode_data["realsense"] = realsense_data

    # Load model realsense data if available
    model_realsense_path = episode_dir / "realsense_model.npz"
    if not model_realsense_path.exists():
        print(f"Warning: model_realsense.npz not found in {episode_dir}")
        model_realsense_data = None
    else:
        model_realsense_data = dict(np.load(model_realsense_path, allow_pickle=True))
        episode_data["model_realsense"] = model_realsense_data

    return episode_data


def main(args):
    data_dirs = args.data_dir
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    output_dir = args.output_dir
    control_mode = args.control_mode

    assert control_mode in [
        "auto",
        "joint",
        "cartesian",
    ], f"Control mode must be 'auto', 'joint', or 'cartesian', got {control_mode}"

    if output_dir is None:
        output_dir = data_dirs[0]
    else:
        os.makedirs(output_dir, exist_ok=True)

    if args.traj_name is None:
        traj_name = time.strftime("%Y%m%d_%H%M%S")
    else:
        traj_name = args.traj_name

    # Auto-detect control mode if needed
    detected_mode = control_mode
    if control_mode == "auto":
        print("Auto-detecting control mode...")
        # Find first episode to detect mode
        first_episode_dir = None
        for data_dir in data_dirs:
            episode_dirs = find_episode_dirs(data_dir)
            if episode_dirs:
                first_episode_dir = episode_dirs[0]
                break

        if first_episode_dir:
            first_episode_data = load_episode_data(first_episode_dir)
            if "action" in first_episode_data:
                detected_mode = detect_control_mode(first_episode_data["action"])
                if detected_mode:
                    print(f"Detected control mode: {detected_mode}")
                else:
                    print(
                        "Warning: Could not auto-detect control mode, defaulting to cartesian"
                    )
                    detected_mode = "cartesian"
            else:
                print(
                    "Warning: No action data found in first episode, defaulting to cartesian"
                )
                detected_mode = "cartesian"
        else:
            print("Warning: No episodes found, defaulting to cartesian")
            detected_mode = "cartesian"

    # Use detected mode for filename and processing
    traj_path = str(
        Path(output_dir).joinpath(f"{traj_name}.{detected_mode}.zarr").absolute()
    )
    replay_buffer = ReplayBuffer.create_from_path(zarr_path=traj_path, mode="w")

    total_episodes = 0
    # Process each data directory
    for data_dir in data_dirs:
        # Find all episode directories
        episode_dirs = find_episode_dirs(data_dir)
        print(f"Found {len(episode_dirs)} episode directories in {data_dir}")

        # Process each episode
        for episode_dir in episode_dirs:
            print(f"Processing {episode_dir}")
            raw_episode_data = load_episode_data(episode_dir)

            n_steps = min(
                len(raw_episode_data["state"]["_buffer_timestamps"]),
                len(raw_episode_data["action"]["_buffer_timestamps"]),
            )
            if "realsense" in raw_episode_data:
                n_steps = min(
                    n_steps,
                    len(raw_episode_data["realsense"]["_buffer_timestamps"]),
                )
            if "model_realsense" in raw_episode_data:
                n_steps = min(
                    n_steps,
                    len(raw_episode_data["model_realsense"]["_buffer_timestamps"]),
                )
            raw_episode_data = dict_apply(raw_episode_data, lambda x: x[:n_steps])

            episode_data = dict(
                obs=dict(),
            )
            for k, v in raw_episode_data["state"].items():
                # use state `timestamp` as episode timestamps
                if k == "_buffer_timestamps":
                    continue
                episode_data["obs"][k] = v

            if "realsense" in raw_episode_data:
                for k, v in raw_episode_data["realsense"].items():
                    if k in ["_buffer_timestamps", "timestamp"]:
                        continue
                    episode_data["obs"][k] = v

                images = {}
                for cam_idx in range(episode_data["obs"]["colors"].shape[1]):
                    images[f"camera_{cam_idx}"] = episode_data["obs"]["colors"][
                        :, cam_idx
                    ]
                episode_data["obs"]["images"] = images
                del episode_data["obs"]["colors"]

                depths = {}
                for cam_idx in range(episode_data["obs"]["depths"].shape[1]):
                    depths[f"camera_{cam_idx}"] = np.expand_dims(
                        episode_data["obs"]["depths"][:, cam_idx], axis=-1
                    )
                episode_data["obs"]["depths"] = depths

            if "model_realsense" in raw_episode_data:
                for k, v in raw_episode_data["model_realsense"].items():
                    if k in ["_buffer_timestamps", "timestamp"]:
                        continue
                    episode_data["obs"][f"model_{k}"] = v

                depths = {}
                for cam_idx in range(episode_data["obs"]["model_depths"].shape[1]):
                    depths[f"camera_{cam_idx}"] = np.expand_dims(
                        episode_data["obs"]["model_depths"][:, cam_idx], axis=-1
                    )
                episode_data["obs"]["model_depths"] = depths

            if detected_mode == "joint":
                episode_data["action"] = np.concatenate(
                    [
                        raw_episode_data["action"]["joint_positions"],
                        raw_episode_data["action"]["gripper_state"],
                    ],
                    axis=-1,
                )
            else:
                episode_data["action"] = np.concatenate(
                    [
                        raw_episode_data["action"]["tcp_position"],
                        raw_episode_data["action"]["tcp_orientation"],
                        raw_episode_data["action"]["gripper_state"],
                    ],
                    axis=-1,
                )
            assert not np.any(
                np.isnan(episode_data["action"])
            ), f"Found NaN values in episode actions with {detected_mode} control mode!"

            replay_buffer.add_episode(episode_data, compressors="disk")
            print(f"Added episode {episode_dir} of {n_steps} steps to replay buffer")
            total_episodes += 1

    print(f"Saved replay buffer with {total_episodes} episodes to {traj_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_dir",
        type=str,
        nargs="+",
        help="Path(s) to the data directory(ies) containing subfolders 'episode_0', 'episode_1', etc.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Path to the output directory"
    )
    parser.add_argument(
        "--control_mode",
        type=str,
        default="auto",
        help="Control mode to process: 'auto' (auto-detect), 'joint', or 'cartesian'",
    )
    parser.add_argument(
        "--traj_name", type=str, default=None, help="Name of the saved zarr file"
    )
    args = parser.parse_args()

    main(args)
