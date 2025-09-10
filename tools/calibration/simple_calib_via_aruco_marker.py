#!/usr/bin/env python3
"""
相机标定程序 - 读取相机，计算内外参并保存
"""

import cv2
import numpy as np
import argparse
import time

try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    REALSENSE_AVAILABLE = False
    print("Warning: pyrealsense2 not available.")

def create_charuco_board():
    """创建Charuco标定板"""
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((8, 5), 0.031, 0.023, charuco_dict)
    return charuco_dict, board

def estimate_pose(image, charuco_dict, intrinsics_matrix, dist_coeffs, board):
    """估计相机到标记的姿态"""
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(charuco_dict, detector_params)
    aruco_detector.detectMarkers(image)

    charucodetector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, _ = charucodetector.detectBoard(image)
    
    print(f"检测到 {len(marker_corners) if marker_corners is not None else 0} 个标记")
    
    if charuco_ids is not None and len(charuco_corners) > 3:
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, board, intrinsics_matrix, dist_coeffs, None, None)
        
        if valid:
            # 构建变换矩阵
            R_target2cam = cv2.Rodrigues(rvec)[0]
            t_target2cam = tvec.reshape(3, 1)
            target2cam = np.eye(4)
            target2cam[:3, :3] = R_target2cam
            target2cam[:3, 3] = t_target2cam.reshape(-1)
            
            # 应用坐标系变换
            local_frame_change_transform = np.zeros((4, 4))
            local_frame_change_transform[1, 0] = 1
            local_frame_change_transform[0, 1] = 1
            local_frame_change_transform[2, 2] = -1
            local_frame_change_transform[3, 3] = 1
            
            cam_to_marker = local_frame_change_transform @ np.linalg.inv(target2cam)
            return cam_to_marker, rvec, tvec
    
    return None, None, None

def calibrate_realsense(camera_serial=None, width=640, height=480, fps=30):
    """使用RealSense相机进行标定"""
    if not REALSENSE_AVAILABLE:
        print("错误: pyrealsense2 未安装")
        return None, None, None, None
    
    try:
        # 获取连接的设备
        context = rs.context()
        devices = context.query_devices()
        
        if len(devices) == 0:
            print("错误: 未检测到RealSense相机")
            return None, None, None, None
        
        # 选择相机
        device = None
        if camera_serial:
            # 使用指定的序列号
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == camera_serial:
                    device = dev
                    break
            if device is None:
                print(f"错误: 未找到序列号为 {camera_serial} 的相机")
                print("可用的相机序列号:")
                for dev in devices:
                    print(f"  - {dev.get_info(rs.camera_info.serial_number)}")
                return None, None, None, None
        else:
            # 使用第一个可用的相机
            device = devices[0]
            camera_serial = device.get_info(rs.camera_info.serial_number)
        
        print(f"使用相机: {device.get_info(rs.camera_info.name)}")
        print(f"序列号: {camera_serial}")
        
        # 初始化相机
        pipeline = rs.pipeline()
        config = rs.config()
        
        # 启用指定的设备
        config.enable_device(camera_serial)
        
        # 配置流
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        
        # 开始流
        profile = pipeline.start(config)
        
        # 获取相机内参
        color_profile = profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        
        # 构建内参矩阵
        intrinsics_matrix = np.array([
            [color_intrinsics.fx, 0, color_intrinsics.ppx],
            [0, color_intrinsics.fy, color_intrinsics.ppy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        # 畸变系数
        dist_coeffs = np.array([
            color_intrinsics.coeffs[0],  # k1
            color_intrinsics.coeffs[1],  # k2
            color_intrinsics.coeffs[2],  # p1
            color_intrinsics.coeffs[3],  # p2
            color_intrinsics.coeffs[4]   # k3
        ], dtype=np.float64)
        
        print(f"相机初始化完成: {width}x{height}@{fps}fps")
        print("预热相机中...")
        
        # 预热1秒
        start_time = time.time()
        while time.time() - start_time < 1.0:
            pipeline.wait_for_frames()
        
        print("开始检测标记...")
        
        # 创建Charuco标定板
        charuco_dict, board = create_charuco_board()
        
        # 获取多帧进行检测
        cam_to_marker = None
        attempts = 0
        max_attempts = 30
        
        while cam_to_marker is None and attempts < max_attempts:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if color_frame:
                # 转换为numpy数组
                color_image = np.asanyarray(color_frame.get_data())
                # RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                # 估计姿态
                cam_to_marker, _, _ = estimate_pose(
                    bgr_image, charuco_dict, intrinsics_matrix, dist_coeffs, board
                )
                
                if cam_to_marker is not None:
                    print("成功检测到标记!")
                    break
            
            attempts += 1
            time.sleep(0.1)
        
        # 停止相机
        pipeline.stop()
        
        if cam_to_marker is None:
            print(f"警告: 未能检测到标记 (尝试了 {attempts} 次)")
            return intrinsics_matrix, dist_coeffs, None, camera_serial
        
        return intrinsics_matrix, dist_coeffs, cam_to_marker, camera_serial
        
    except Exception as e:
        print(f"相机标定出错: {e}")
        return None, None, None, None

def main():
    parser = argparse.ArgumentParser(description="相机标定 - 获取内外参")
    parser.add_argument("--serial", type=str, default=None, help="相机序列号 (可选，不指定则使用第一个可用相机)")
    parser.add_argument("--list", action="store_true", help="列出所有可用的相机")
    parser.add_argument("--width", type=int, default=640, help="相机宽度")
    parser.add_argument("--height", type=int, default=480, help="相机高度")
    parser.add_argument("--fps", type=int, default=30, help="相机帧率")
    
    args = parser.parse_args()
    
    # 如果请求列出相机
    if args.list:
        if not REALSENSE_AVAILABLE:
            print("错误: pyrealsense2 未安装")
            return
        
        context = rs.context()
        devices = context.query_devices()
        
        if len(devices) == 0:
            print("未检测到RealSense相机")
        else:
            print("可用的RealSense相机:")
            for i, dev in enumerate(devices):
                print(f"  {i+1}. {dev.get_info(rs.camera_info.name)}")
                print(f"     序列号: {dev.get_info(rs.camera_info.serial_number)}")
                if dev.get_info(rs.camera_info.firmware_version):
                    print(f"     固件版本: {dev.get_info(rs.camera_info.firmware_version)}")
        return
    
    print("开始标定相机...")
    
    # 执行标定
    intrinsics, dist_coeffs, extrinsics, camera_serial = calibrate_realsense(
        args.serial, args.width, args.height, args.fps
    )
    
    if intrinsics is None:
        print("标定失败")
        return
    
    # 准备保存的数据
    calibration_data = {
        'camera_serial': camera_serial,
        'intrinsics': intrinsics,
        'dist_coeffs': dist_coeffs,
        'extrinsics': extrinsics,  # cam_to_marker transformation
        'image_size': (args.width, args.height),
        'timestamp': time.time()
    }
    
    # 保存文件 - 使用相机序列号命名
    output_file = f"{camera_serial}.npy"
    np.save(output_file, calibration_data, allow_pickle=True)
    print(f"\n标定数据已保存至: {output_file}")
    
    # 打印结果
    print("\n==================== 标定结果 ====================")
    print(f"相机序列号: {camera_serial}")
    print(f"图像尺寸: {args.width}x{args.height}")
    
    print("\n内参矩阵:")
    print(intrinsics)
    
    print("\n畸变系数:")
    print(dist_coeffs)
    
    if extrinsics is not None:
        print("\n外参矩阵 (相机到标记的变换):")
        print(extrinsics)
        
        # 提取平移和旋转
        translation = extrinsics[:3, 3]
        from scipy.spatial.transform import Rotation as R
        rotation = R.from_matrix(extrinsics[:3, :3])
        euler_angles = rotation.as_euler('xyz', degrees=True)
        
        print(f"\n平移 (m):")
        print(f"  X: {translation[0]:.4f}")
        print(f"  Y: {translation[1]:.4f}")
        print(f"  Z: {translation[2]:.4f}")
        
        print(f"\n旋转 (度):")
        print(f"  Roll:  {euler_angles[0]:.2f}")
        print(f"  Pitch: {euler_angles[1]:.2f}")
        print(f"  Yaw:   {euler_angles[2]:.2f}")
    else:
        print("\n外参: 未检测到标记，无法计算外参")
    
    print("=" * 50)

if __name__ == "__main__":
    main()