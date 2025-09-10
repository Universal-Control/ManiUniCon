#!/usr/bin/env python3
"""
Multi-camera point cloud real-time visualization program
Reads saved calibration results, transforms point clouds from any number of cameras to marker coordinate system, and visualizes them in real-time using meshcat
"""

import numpy as np
import argparse
import time
import threading
from pathlib import Path
from typing import List, Dict, Any

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available.")

try:
    import meshcat
    import meshcat.geometry as g
    import meshcat.transformations as tf
    MESHCAT_AVAILABLE = True
except ImportError:
    MESHCAT_AVAILABLE = False
    print("Warning: meshcat not available. Install with: pip install meshcat")


class MultiCameraPointCloudVisualizer:
    """Multi-camera point cloud visualizer"""
    
    def __init__(self, calib_files: List[str], max_points: int = 10000, 
                 voxel_size: float = 0.005, max_depth: float = 2.0):
        """
        Initialize visualizer
        
        Args:
            calib_files: List of camera calibration file paths
            max_points: Maximum number of points per point cloud (for downsampling)
            voxel_size: Voxel size (for downsampling)
            max_depth: Maximum depth threshold (meters)
        """
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not available")
        if not MESHCAT_AVAILABLE:
            raise RuntimeError("meshcat not available")
        
        # Load all calibration data
        self.calibrations = []
        self.cameras = []
        
        for i, calib_file in enumerate(calib_files):
            calib_data = np.load(calib_file, allow_pickle=True).item()
            
            # Check if extrinsics exist
            if calib_data['extrinsics'] is None:
                print(f"Warning: Camera {calib_data['camera_serial']} has no extrinsic data, skipping")
                continue
            
            self.calibrations.append(calib_data)
            print(f"Loading camera {i+1} calibration: {calib_data['camera_serial']}")
        
        if len(self.calibrations) == 0:
            raise ValueError("No valid camera calibration data")
        
        self.num_cameras = len(self.calibrations)
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        
        # Assign colors for each camera (to distinguish point clouds from different cameras)
        self.camera_colors = self.generate_colors(self.num_cameras)
        
        # Initialize meshcat
        self.vis = meshcat.Visualizer()
        self.vis.open()
        print(f"Meshcat server started: {self.vis.url()}")
        
        # Setup coordinate frame visualization
        self.setup_coordinate_frames()
        
        # Thread control
        self.running = False
        self.viz_thread = None
        self.lock = threading.Lock()
    
    def generate_colors(self, n: int) -> List[np.ndarray]:
        """Generate different colors for n cameras"""
        # Use HSV color space to generate evenly distributed colors
        colors = []
        for i in range(n):
            hue = i / n
            # HSV to RGB conversion
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(np.array(rgb))
        return colors
    
    def setup_coordinate_frames(self):
        """Setup coordinate frame visualization"""
        # Marker coordinate system (world coordinate system)
        self.vis["frames"]["marker"].set_object(
            g.triad(scale=0.1)
        )
        
        # Setup coordinate system for each camera
        for i, calib in enumerate(self.calibrations):
            camera_name = f"camera{i+1}"
            
            # Camera coordinate system
            self.vis["frames"][camera_name].set_object(
                g.triad(scale=0.05)
            )
            
            # Set camera position in marker coordinate system
            marker_to_cam = np.linalg.inv(calib['extrinsics'])
            self.vis["frames"][camera_name].set_transform(marker_to_cam)
            
            # Add label (represented by small sphere)
            label_pos = marker_to_cam[:3, 3] + np.array([0, 0, 0.05])
            self.add_label(f"labels/{camera_name}", label_pos, self.camera_colors[i])
        
        # Add marker label
        self.add_label("labels/marker", [0, 0, 0.6], [1, 1, 1])
    
    def add_label(self, path: str, position: np.ndarray, color: List[float]):
        """Add label (represented by colored sphere)"""
        color_hex = int(color[0]*255) << 16 | int(color[1]*255) << 8 | int(color[2]*255)
        self.vis[path].set_object(
            g.Sphere(radius=0.01),
            g.MeshLambertMaterial(color=color_hex)
        )
        self.vis[path].set_transform(
            tf.translation_matrix(position)
        )
    
    def init_cameras(self, width, height):
        """Initialize all RealSense cameras"""
        context = rs.context()
        devices = context.query_devices()
        available_serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        
        print(f"Detected {len(devices)} cameras: {available_serials}")
        
        # Check if all required cameras are available
        for calib in self.calibrations:
            if calib['camera_serial'] not in available_serials:
                raise RuntimeError(f"Camera {calib['camera_serial']} not connected")
        
        # Initialize each camera
        for i, calib in enumerate(self.calibrations):
            camera_info = {
                'index': i,
                'serial': calib['camera_serial'],
                'pipeline': rs.pipeline(),
                'align': None,
                'depth_scale': None,
                'intrinsics': None,
                'extrinsics': calib['extrinsics'],
                'width': width,
                'height': height
            }
            
            # Configure camera
            config = rs.config()
            config.enable_device(camera_info['serial'])
            config.enable_stream(rs.stream.depth, 
                               camera_info['width'], 
                               camera_info['height'], 
                               rs.format.z16, 30)
            config.enable_stream(rs.stream.color, 
                               camera_info['width'], 
                               camera_info['height'], 
                               rs.format.rgb8, 30)
            
            # Start camera
            profile = camera_info['pipeline'].start(config)
            camera_info['align'] = rs.align(rs.stream.color)
            color_profile = profile.get_stream(rs.stream.color)
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            camera_info['intrinsics'] = np.array([
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float64)
            # Get depth scale factor
            depth_sensor = profile.get_device().first_depth_sensor()
            camera_info['depth_scale'] = depth_sensor.get_depth_scale()
            
            self.cameras.append(camera_info)
            print(f"Camera {i+1} ({camera_info['serial']}) initialization complete")
    
    def depth_to_pointcloud(self, color_image: np.ndarray, depth_image: np.ndarray, 
                          intrinsics: np.ndarray, depth_scale: float) -> tuple:
        """Convert depth map to point cloud"""
        height, width = depth_image.shape
        
        # Create pixel coordinate grid
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # Convert depth values to meters
        z = depth_image * depth_scale
        
        # Filter invalid depths
        valid_mask = (z > 0) & (z < self.max_depth)
        
        # Camera intrinsics
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        # Calculate 3D coordinates
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        
        # Extract valid points
        points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)
        colors = color_image[valid_mask] / 255.0
        
        # Downsample
        if len(points) > self.max_points:
            # Random sampling
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        return points, colors
    
    def transform_pointcloud(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """Transform point cloud to new coordinate system"""
        # Add homogeneous coordinates
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        # Apply transformation
        points_transformed = (transform @ points_h.T).T
        return points_transformed[:, :3]
    
    def update_visualization(self):
        """Update visualization"""
        for i, camera in enumerate(self.cameras):
            try:
                # Get camera data
                frames = camera['pipeline'].wait_for_frames(timeout_ms=500)
                aligned_frames = camera['align'].process(frames)
                print(f"Get frame from camera {i}")
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if depth_frame and color_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # Generate point cloud
                    points, colors = self.depth_to_pointcloud(
                        color_image, depth_image, 
                        camera['intrinsics'], camera['depth_scale']
                    )
                    
                    # Transform to marker coordinate system
                    if len(points) > 0:
                        points_marker = self.transform_pointcloud(points, camera['extrinsics'])
                        
                        # Choose whether to use original colors or camera-specific colors
                        use_original_colors = True  # Can be set to False to use camera-specific colors
                        
                        if use_original_colors:
                            point_colors = colors.T
                        else:
                            # Use camera-specific colors
                            camera_color = self.camera_colors[camera['index']]
                            point_colors = np.tile(camera_color.reshape(-1, 1), (1, len(points)))
                        
                        # Update point cloud in meshcat
                        self.vis[f"pointcloud{camera['index']+1}"].set_object(
                            g.PointCloud(
                                position=points_marker.T,
                                color=point_colors,
                                size=0.002
                            )
                        )
                        
            except Exception as e:
                # Timeout or other error, ignore this frame
                print(f"Error: {e}")
                pass
    
    def visualization_loop(self):
        """Visualization loop thread"""
        print("Starting real-time visualization...")
        frame_count = 0
        start_time = time.time()
        
        while self.running:
            with self.lock:
                self.update_visualization()
            
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed = time.time() - start_time
                fps = 30 / elapsed if elapsed > 0 else 0
                print(f"FPS: {fps:.1f}")
                start_time = time.time()
            
            time.sleep(0.01)  # Approximately 100Hz update rate
    
    def start(self, width, height):
        """Start visualization"""
        try:
            # Initialize cameras
            self.init_cameras(width, height)
            
            # # Warm up cameras
            # print("Warming up cameras...")
            # for _ in range(6):
            #     for camera in self.cameras:
            #         print(f"Warming up camera {camera['serial']}...")
            #         camera['pipeline'].wait_for_frames()
            #         time.sleep(1)
                
            
            # Start visualization thread
            self.running = True
            self.viz_thread = threading.Thread(target=self.visualization_loop)
            self.viz_thread.start( )
            
            print("\nVisualization started!")
            print("Open the following link in browser to view:")
            print(f"  {self.vis.url()}")
            print(f"\nVisualizing point clouds from {self.num_cameras} cameras")
            print("Press Ctrl+C to stop...")
            
            # Main thread wait
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\nStopping...")
            self.stop()
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
    
    def stop(self):
        """Stop visualization"""
        self.running = False
        
        if self.viz_thread:
            self.viz_thread.join(timeout=2.0)
        
        for camera in self.cameras:
            camera['pipeline'].stop()
            print(f"Camera {camera['serial']} stopped")
        
        print("Visualization stopped")


def main():
    parser = argparse.ArgumentParser(description="Multi-camera point cloud real-time visualization")
    parser.add_argument("calib_files", nargs='+', type=str, 
                       help="List of camera calibration files (.npy)")
    parser.add_argument("--max_points", type=int, default=200000, 
                       help="Maximum number of points per point cloud")
    parser.add_argument("--voxel_size", type=float, default=0.001, 
                       help="Voxel size (meters)")
    parser.add_argument("--max_depth", type=float, default=3.0,
                       help="Maximum depth threshold (meters)")
    parser.add_argument("--width", type=int, default=640, 
                       help="Camera width")
    parser.add_argument("--height", type=int, default=480, 
                       help="Camera height")
    args = parser.parse_args()
    
    # Check if files exist
    for calib_file in args.calib_files:
        if not Path(calib_file).exists():
            print(f"Error: Calibration file does not exist: {calib_file}")
            return
    
    print(f"Preparing to visualize {len(args.calib_files)} cameras")
    
    # Create visualizer
    visualizer = MultiCameraPointCloudVisualizer(
        args.calib_files,
        max_points=args.max_points,
        voxel_size=args.voxel_size,
        max_depth=args.max_depth
    )
    
    # Start visualization
    visualizer.start(args.width, args.height)


if __name__ == "__main__":
    main()