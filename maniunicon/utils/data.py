import os


def get_next_episode_dir(base_dir: str) -> str:
    """Get the next episode directory path with incremental numbering.

    Args:
        base_dir: Base directory for recording

    Returns:
        Path to the next episode directory (e.g., "episode_0", "episode_1", etc.)
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    episode_num = 0
    while True:
        episode_dir = os.path.join(base_dir, f"episode_{episode_num}")
        if not os.path.exists(episode_dir):
            os.makedirs(episode_dir)
            return episode_dir
        episode_num += 1
