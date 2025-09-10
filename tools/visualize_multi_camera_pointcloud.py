#!/usr/bin/env python3
"""
多相机点云实时可视化程序
读取保存的标定结果，将任意数量相机的点云转换到标记坐标系下，并用meshcat实时可视化
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
    """多相机点云可视化器"""
    
    def __init__(self, calib_files: List[str], max_points: int = 10000, 
                 voxel_size: float = 0.005, max_depth: float = 2.0):
        """
        初始化可视化器
        
        Args:
            calib_files: 相机标定文件路径列表
            max_points: 每个点云的最大点数（用于下采样）
            voxel_size: 体素大小（用于下采样）
            max_depth: 最大深度阈值（米）
        """
        if not REALSENSE_AVAILABLE:
            raise RuntimeError("pyrealsense2 not available")
        if not MESHCAT_AVAILABLE:
            raise RuntimeError("meshcat not available")
        
        # 加载所有标定数据
        self.calibrations = []
        self.cameras = []
        
        for i, calib_file in enumerate(calib_files):
            calib_data = np.load(calib_file, allow_pickle=True).item()
            
            # 检查外参是否存在
            if calib_data['extrinsics'] is None:
                print(f"警告: 相机 {calib_data['camera_serial']} 没有外参数据，跳过")
                continue
            
            self.calibrations.append(calib_data)
            print(f"加载相机{i+1}标定: {calib_data['camera_serial']}")
        
        if len(self.calibrations) == 0:
            raise ValueError("没有有效的相机标定数据")
        
        self.num_cameras = len(self.calibrations)
        self.max_points = max_points
        self.voxel_size = voxel_size
        self.max_depth = max_depth
        
        # 为每个相机分配颜色（用于区分不同相机的点云）
        self.camera_colors = self.generate_colors(self.num_cameras)
        
        # 初始化meshcat
        self.vis = meshcat.Visualizer()
        self.vis.open()
        print(f"Meshcat服务器已启动: {self.vis.url()}")
        
        # 设置坐标系可视化
        self.setup_coordinate_frames()
        
        # 线程控制
        self.running = False
        self.viz_thread = None
        self.lock = threading.Lock()
    
    def generate_colors(self, n: int) -> List[np.ndarray]:
        """为n个相机生成不同的颜色"""
        # 使用HSV色彩空间生成均匀分布的颜色
        colors = []
        for i in range(n):
            hue = i / n
            # HSV到RGB转换
            import colorsys
            rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
            colors.append(np.array(rgb))
        return colors
    
    def setup_coordinate_frames(self):
        """设置坐标系可视化"""
        # 标记坐标系（世界坐标系）
        self.vis["frames"]["marker"].set_object(
            g.triad(scale=0.1)
        )
        
        # 为每个相机设置坐标系
        for i, calib in enumerate(self.calibrations):
            camera_name = f"camera{i+1}"
            
            # 相机坐标系
            self.vis["frames"][camera_name].set_object(
                g.triad(scale=0.05)
            )
            
            # 设置相机在标记坐标系下的位置
            marker_to_cam = np.linalg.inv(calib['extrinsics'])
            self.vis["frames"][camera_name].set_transform(marker_to_cam)
            
            # 添加标签（用小球表示）
            label_pos = marker_to_cam[:3, 3] + np.array([0, 0, 0.05])
            self.add_label(f"labels/{camera_name}", label_pos, self.camera_colors[i])
        
        # 添加标记标签
        self.add_label("labels/marker", [0, 0, 0.6], [1, 1, 1])
    
    def add_label(self, path: str, position: np.ndarray, color: List[float]):
        """添加标签（用彩色小球表示）"""
        color_hex = int(color[0]*255) << 16 | int(color[1]*255) << 8 | int(color[2]*255)
        self.vis[path].set_object(
            g.Sphere(radius=0.01),
            g.MeshLambertMaterial(color=color_hex)
        )
        self.vis[path].set_transform(
            tf.translation_matrix(position)
        )
    
    def init_cameras(self, width, height):
        """初始化所有RealSense相机"""
        context = rs.context()
        devices = context.query_devices()
        available_serials = [dev.get_info(rs.camera_info.serial_number) for dev in devices]
        
        print(f"检测到 {len(devices)} 个相机: {available_serials}")
        
        # 检查所需相机是否都可用
        for calib in self.calibrations:
            if calib['camera_serial'] not in available_serials:
                raise RuntimeError(f"相机 {calib['camera_serial']} 未连接")
        
        # 初始化每个相机
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
            
            # 配置相机
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
            
            # 启动相机
            profile = camera_info['pipeline'].start(config)
            camera_info['align'] = rs.align(rs.stream.color)
            color_profile = profile.get_stream(rs.stream.color)
            color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
            camera_info['intrinsics'] = np.array([
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1]
            ], dtype=np.float64)
            # 获取深度缩放因子
            depth_sensor = profile.get_device().first_depth_sensor()
            camera_info['depth_scale'] = depth_sensor.get_depth_scale()
            
            self.cameras.append(camera_info)
            print(f"相机{i+1} ({camera_info['serial']}) 初始化完成")
    
    def depth_to_pointcloud(self, color_image: np.ndarray, depth_image: np.ndarray, 
                          intrinsics: np.ndarray, depth_scale: float) -> tuple:
        """将深度图转换为点云"""
        height, width = depth_image.shape
        
        # 创建像素坐标网格
        xx, yy = np.meshgrid(np.arange(width), np.arange(height))
        
        # 转换深度值到米
        z = depth_image * depth_scale
        
        # 过滤无效深度
        valid_mask = (z > 0) & (z < self.max_depth)
        
        # 相机内参
        fx = intrinsics[0, 0]
        fy = intrinsics[1, 1]
        cx = intrinsics[0, 2]
        cy = intrinsics[1, 2]
        
        # 计算3D坐标
        x = (xx - cx) * z / fx
        y = (yy - cy) * z / fy
        
        # 提取有效点
        points = np.stack([x[valid_mask], y[valid_mask], z[valid_mask]], axis=-1)
        colors = color_image[valid_mask] / 255.0
        
        # 下采样
        if len(points) > self.max_points:
            # 随机采样
            indices = np.random.choice(len(points), self.max_points, replace=False)
            points = points[indices]
            colors = colors[indices]
        
        return points, colors
    
    def transform_pointcloud(self, points: np.ndarray, transform: np.ndarray) -> np.ndarray:
        """将点云转换到新坐标系"""
        # 添加齐次坐标
        points_h = np.hstack([points, np.ones((points.shape[0], 1))])
        # 应用变换
        points_transformed = (transform @ points_h.T).T
        return points_transformed[:, :3]
    
    def update_visualization(self):
        """更新可视化"""
        for i, camera in enumerate(self.cameras):
            try:
                # 获取相机数据
                frames = camera['pipeline'].wait_for_frames(timeout_ms=500)
                aligned_frames = camera['align'].process(frames)
                print(f"Get frame from camera {i}")
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if depth_frame and color_frame:
                    depth_image = np.asanyarray(depth_frame.get_data())
                    color_image = np.asanyarray(color_frame.get_data())
                    
                    # 生成点云
                    points, colors = self.depth_to_pointcloud(
                        color_image, depth_image, 
                        camera['intrinsics'], camera['depth_scale']
                    )
                    
                    # 转换到标记坐标系
                    if len(points) > 0:
                        points_marker = self.transform_pointcloud(points, camera['extrinsics'])
                        
                        # 选择是否使用原始颜色或相机特定颜色
                        use_original_colors = True  # 可以设置为False来使用相机特定颜色
                        
                        if use_original_colors:
                            point_colors = colors.T
                        else:
                            # 使用相机特定颜色
                            camera_color = self.camera_colors[camera['index']]
                            point_colors = np.tile(camera_color.reshape(-1, 1), (1, len(points)))
                        
                        # 更新meshcat中的点云
                        self.vis[f"pointcloud{camera['index']+1}"].set_object(
                            g.PointCloud(
                                position=points_marker.T,
                                color=point_colors,
                                size=0.002
                            )
                        )
                        
            except Exception as e:
                # 超时或其他错误，忽略这一帧
                print(f"错误: {e}")
                pass
    
    def visualization_loop(self):
        """可视化循环线程"""
        print("开始实时可视化...")
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
            
            time.sleep(0.01)  # 约100Hz更新率
    
    def start(self, width, height):
        """启动可视化"""
        try:
            # 初始化相机
            self.init_cameras(width, height)
            
            # # 预热相机
            # print("预热相机中...")
            # for _ in range(6):
            #     for camera in self.cameras:
            #         print(f"预热相机 {camera['serial']}...")
            #         camera['pipeline'].wait_for_frames()
            #         time.sleep(1)
                
            
            # 启动可视化线程
            self.running = True
            self.viz_thread = threading.Thread(target=self.visualization_loop)
            self.viz_thread.start( )
            
            print("\n可视化已启动!")
            print("在浏览器中打开以下链接查看:")
            print(f"  {self.vis.url()}")
            print(f"\n正在可视化 {self.num_cameras} 个相机的点云")
            print("按 Ctrl+C 停止...")
            
            # 主线程等待
            while self.running:
                time.sleep(0.1)
                
        except KeyboardInterrupt:
            print("\n正在停止...")
            self.stop()
        except Exception as e:
            print(f"错误: {e}")
            import traceback
            traceback.print_exc()
            self.stop()
    
    def stop(self):
        """停止可视化"""
        self.running = False
        
        if self.viz_thread:
            self.viz_thread.join(timeout=2.0)
        
        for camera in self.cameras:
            camera['pipeline'].stop()
            print(f"相机 {camera['serial']} 已停止")
        
        print("可视化已停止")


def main():
    parser = argparse.ArgumentParser(description="多相机点云实时可视化")
    parser.add_argument("calib_files", nargs='+', type=str, 
                       help="相机标定文件列表 (.npy)")
    parser.add_argument("--max_points", type=int, default=200000, 
                       help="每个点云的最大点数")
    parser.add_argument("--voxel_size", type=float, default=0.001, 
                       help="体素大小（米）")
    parser.add_argument("--max_depth", type=float, default=3.0,
                       help="最大深度阈值（米）")
    parser.add_argument("--width", type=int, default=640, 
                       help="相机宽度")
    parser.add_argument("--height", type=int, default=480, 
                       help="相机高度")
    args = parser.parse_args()
    
    # 检查文件是否存在
    for calib_file in args.calib_files:
        if not Path(calib_file).exists():
            print(f"错误: 标定文件不存在: {calib_file}")
            return
    
    print(f"准备可视化 {len(args.calib_files)} 个相机")
    
    # 创建可视化器
    visualizer = MultiCameraPointCloudVisualizer(
        args.calib_files,
        max_points=args.max_points,
        voxel_size=args.voxel_size,
        max_depth=args.max_depth
    )
    
    # 启动可视化
    visualizer.start(args.width, args.height)


if __name__ == "__main__":
    main()