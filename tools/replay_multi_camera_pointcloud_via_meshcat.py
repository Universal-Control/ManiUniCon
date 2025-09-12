#!/usr/bin/env python3
"""
Multi-camera point cloud replay visualization program
Reads saved NPZ data and visualizes point clouds in meshcat
"""

import numpy as np
import argparse
import time
import threading
from pathlib import Path
from typing import List

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf

    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    print("Warning: meshcat not available. Install with: pip install meshcat")


class ReplayPointCloudVisualizer:
    """Replay point cloud visualizer from saved NPZ data"""

    def __init__(
        self,
        npz_file: str,
        max_points: int = 10000,
        voxel_size: float = 0.005,
        max_depth: float = 2.0,
    ):
        """
        Initialize visualizer

        Args:
            npz_file: Path to NPZ file containing saved camera data
            max_points: Maximum number of points per point cloud (for downsampling)
            voxel_size: Voxel size (for downsampling)
            max_depth: Maximum depth threshold (meters)
        """
        if not MESHCAT_AVAILABLE:
            raise RuntimeError("meshcat not available")

        # Load saved data
        print(f"Loading data from {npz_file}")
        self.data = np.load(npz_file)

        # Extract data - handle multiple cameras
        self.color_images = self.data[
            "colors"
        ]  # (T, N, H, W, 3) - T: frames, N: cameras
        self.depth_images = self.data["depths"]  # (T, N, H, W)
        self.intrinsics = self.data[
            "intrs"
        ]  # (T, N, 4) - fx, fy, cx, cy for each camera at each frame
        self.extrinsics = self.data[
            "transforms"
        ]  # (T, N, 4, 4) - transforms for each camera at each frame

        self.num_frames = self.color_images.shape[0]
        self.num_cameras = self.color_images.shape[1]
        self.height = self.color_images.shape[2]
        self.width = self.color_images.shape[3]

        print(f"Loaded {self.num_frames} frames from {self.num_cameras} camera(s)")
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Visualizing all {self.num_cameras} cameras merged")
        print(f"First frame extrinsics shape: {self.extrinsics[0].shape}")

        self.max_points = max_points
        self.voxel_size = voxel_size
        self.max_depth = max_depth

        # Initialize meshcat
        self.vis = meshcat.Visualizer()
        self.vis.open()
        print(f"Meshcat server started: {self.vis.url()}")

        # Setup coordinate frame visualization
        self.setup_coordinate_frames()

        # Playback control
        self.current_frame = 0
        self.playing = False
        self.fps = 30
        self.lock = threading.Lock()

    def setup_coordinate_frames(self):
        """Setup coordinate frame visualization"""
        # World coordinate system
        self.vis["frames"]["world"].set_object(g.triad(scale=0.1))

        # Setup coordinate frames for all cameras
        for cam_idx in range(self.num_cameras):
            # Camera coordinate system
            self.vis["frames"][f"camera_{cam_idx}"].set_object(g.triad(scale=0.05))

            # Set camera position using extrinsics (camera to world transform)
            if self.extrinsics is not None:
                # The extrinsics is the camera-to-world transform for first frame
                first_frame_extr = self.extrinsics[0, cam_idx]
                self.vis["frames"][f"camera_{cam_idx}"].set_transform(first_frame_extr)

                # Add camera label with different colors
                camera_pos = first_frame_extr[:3, 3]
                color = self.get_camera_color(cam_idx)
                self.add_label(
                    f"labels/camera_{cam_idx}",
                    camera_pos + np.array([0, 0, 0.05]),
                    color,
                )

        # Add world label
        self.add_label("labels/world", [0, 0, 0.1], [1, 0, 0])

    def get_camera_color(self, cam_idx):
        """Get color for camera visualization"""
        colors = [[0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        return colors[cam_idx % len(colors)]

    def add_label(self, path: str, position: np.ndarray, color: List[float]):
        """Add label (represented by colored sphere)"""
        color_hex = (
            int(color[0] * 255) << 16 | int(color[1] * 255) << 8 | int(color[2] * 255)
        )
        self.vis[path].set_object(
            g.Sphere(radius=0.01), g.MeshLambertMaterial(color=color_hex)
        )
        self.vis[path].set_transform(tf.translation_matrix(position))

    def depth_to_pointcloud(
        self,
        color_image: np.ndarray,
        depth_image: np.ndarray,
        frame_idx: int,
        cam_idx: int,
    ) -> tuple:
        """Convert depth map to point cloud for a specific camera"""
        height, width = depth_image.shape

        # Create pixel coordinate grid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))

        # Depth is already in meters from the saved data
        z = depth_image

        # Filter invalid depths
        valid_mask = (z > 0) & (z < self.max_depth)

        # Camera intrinsics for current frame and camera
        camera_intrinsics = self.intrinsics[frame_idx, cam_idx]
        fx = camera_intrinsics[0]
        fy = camera_intrinsics[1]
        cx = camera_intrinsics[2]
        cy = camera_intrinsics[3]

        # Calculate 3D coordinates
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy

        # Extract valid points
        points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)
        colors = color_image[valid_mask] / 255.0

        return points, colors

    def transform_pointcloud(
        self, points: np.ndarray, transform: np.ndarray
    ) -> np.ndarray:
        """Transform point cloud to new coordinate system"""
        # Add homogeneous coordinates
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        # Apply transformation
        points_transformed = (transform @ points_h.T).T
        return points_transformed[:, :3]

    def update_visualization(self):
        """Update visualization with current frame - merge all cameras"""
        if self.current_frame >= self.num_frames:
            self.current_frame = 0  # Loop back to beginning

        # Collect point clouds from all cameras
        all_points = []
        all_colors = []

        for cam_idx in range(self.num_cameras):
            # Get current frame data for this camera
            color_image = self.color_images[self.current_frame, cam_idx]
            depth_image = self.depth_images[self.current_frame, cam_idx]

            # Generate point cloud
            points, colors = self.depth_to_pointcloud(
                color_image, depth_image, self.current_frame, cam_idx
            )

            # Transform to world coordinate system if extrinsics available
            if len(points) > 0:
                if self.extrinsics is not None:
                    # Get extrinsics for current frame and camera
                    current_extrinsics = self.extrinsics[self.current_frame, cam_idx]
                    points_world = self.transform_pointcloud(points, current_extrinsics)
                else:
                    points_world = points

                all_points.append(points_world)
                all_colors.append(colors)

        # Merge all point clouds
        if len(all_points) > 0:
            merged_points = np.vstack(all_points)
            merged_colors = np.vstack(all_colors)

            # Downsample if too many points
            if len(merged_points) > self.max_points:
                indices = np.random.choice(
                    len(merged_points), self.max_points, replace=False
                )
                merged_points = merged_points[indices]
                merged_colors = merged_colors[indices]

            # Update point cloud in meshcat
            self.vis["pointcloud"].set_object(
                g.PointCloud(
                    position=merged_points.T, color=merged_colors.T, size=0.002
                )
            )

        # Print frame info
        if self.current_frame % 10 == 0:
            print(f"Frame {self.current_frame + 1}/{self.num_frames}")

    def playback_loop(self):
        """Playback loop thread"""
        print("Starting replay...")

        while self.playing:
            with self.lock:
                self.update_visualization()

                # Advance to next frame
                self.current_frame += 1
                if self.current_frame >= self.num_frames:
                    if self.loop:
                        self.current_frame = 0
                    else:
                        self.playing = False
                        print("Playback finished")

            # Control playback speed
            time.sleep(1.0 / self.fps)

    def start(self, loop=True, auto_play=True):
        """Start visualization

        Args:
            loop: Whether to loop playback
            auto_play: Whether to start playing automatically
        """
        try:
            self.loop = loop

            print("\nVisualization started!")
            print("Open the following link in browser to view:")
            print(f"  {self.vis.url()}")
            print(f"\nReplaying {self.num_frames} frames")
            print("\nControls:")
            print("  Space: Play/Pause")
            print("  R: Reset to first frame")
            print("  L: Toggle loop mode")
            print("  Q/Ctrl+C: Quit")
            print("  Left/Right arrows: Previous/Next frame")
            print("  Up/Down arrows: Increase/Decrease playback speed")

            # Display first frame
            self.update_visualization()

            if auto_play:
                self.play()

            # Interactive control loop
            self.interactive_control()

        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()
        except Exception as e:
            print(f"Error: {e}")
            import traceback

            traceback.print_exc()
            self.stop()

    def play(self):
        """Start playback"""
        if not self.playing:
            self.playing = True
            self.playback_thread = threading.Thread(target=self.playback_loop)
            self.playback_thread.start()
            print("Playing...")

    def pause(self):
        """Pause playback"""
        if self.playing:
            self.playing = False
            if hasattr(self, "playback_thread"):
                self.playback_thread.join()
            print("Paused")

    def reset(self):
        """Reset to first frame"""
        with self.lock:
            self.current_frame = 0
            self.update_visualization()
            print("Reset to first frame")

    def next_frame(self):
        """Go to next frame"""
        with self.lock:
            self.current_frame = (self.current_frame + 1) % self.num_frames
            self.update_visualization()

    def prev_frame(self):
        """Go to previous frame"""
        with self.lock:
            self.current_frame = (self.current_frame - 1) % self.num_frames
            self.update_visualization()

    def increase_speed(self):
        """Increase playback speed"""
        self.fps = min(self.fps * 1.5, 120)
        print(f"Playback speed: {self.fps:.1f} fps")

    def decrease_speed(self):
        """Decrease playback speed"""
        self.fps = max(self.fps / 1.5, 1)
        print(f"Playback speed: {self.fps:.1f} fps")

    def interactive_control(self):
        """Interactive keyboard control"""
        try:
            import sys, tty, termios

            # Get terminal settings
            old_settings = termios.tcgetattr(sys.stdin)

            try:
                # Set terminal to raw mode
                tty.setraw(sys.stdin.fileno())

                while True:
                    # Read single character
                    key = sys.stdin.read(1)

                    if key == " ":  # Space - play/pause
                        if self.playing:
                            self.pause()
                        else:
                            self.play()
                    elif key == "r" or key == "R":  # Reset
                        self.reset()
                    elif key == "l" or key == "L":  # Toggle loop
                        self.loop = not self.loop
                        print(f"\nLoop mode: {'ON' if self.loop else 'OFF'}")
                    elif (
                        key == "q" or key == "Q" or key == "\x03"
                    ):  # Quit (q or Ctrl+C)
                        break
                    elif key == "\x1b":  # Escape sequence (arrow keys)
                        next1, next2 = sys.stdin.read(2)
                        if next1 == "[":
                            if next2 == "C":  # Right arrow
                                self.next_frame()
                            elif next2 == "D":  # Left arrow
                                self.prev_frame()
                            elif next2 == "A":  # Up arrow
                                self.increase_speed()
                            elif next2 == "B":  # Down arrow
                                self.decrease_speed()

            finally:
                # Restore terminal settings
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        except ImportError:
            # Fallback for systems without termios
            print("\nInteractive controls not available. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                pass

    def stop(self):
        """Stop visualization"""
        self.playing = False

        if hasattr(self, "playback_thread"):
            self.playback_thread.join(timeout=2.0)

        print("Visualization stopped")


def main():
    parser = argparse.ArgumentParser(
        description="Replay point cloud from saved NPZ file"
    )
    parser.add_argument(
        "--npz_file",
        type=str,
        default="output/episode_0/realsense.npz",
        help="Path to NPZ file containing saved camera data",
    )
    parser.add_argument(
        "--max_points",
        type=int,
        default=200000,
        help="Maximum number of points per point cloud",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.001, help="Voxel size (meters)"
    )
    parser.add_argument(
        "--max_depth", type=float, default=3.0, help="Maximum depth threshold (meters)"
    )
    parser.add_argument(
        "--loop", action="store_true", default=True, help="Loop playback"
    )
    parser.add_argument(
        "--auto_play",
        action="store_true",
        default=True,
        help="Start playing automatically",
    )
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS")
    args = parser.parse_args()

    # Check if file exists
    if not Path(args.npz_file).exists():
        print(f"Error: NPZ file does not exist: {args.npz_file}")
        return

    print(f"Loading data from: {args.npz_file}")

    # Create visualizer
    visualizer = ReplayPointCloudVisualizer(
        args.npz_file,
        max_points=args.max_points,
        voxel_size=args.voxel_size,
        max_depth=args.max_depth,
    )

    # Set playback FPS
    visualizer.fps = args.fps

    # Start visualization
    visualizer.start(loop=args.loop, auto_play=args.auto_play)


if __name__ == "__main__":
    main()
