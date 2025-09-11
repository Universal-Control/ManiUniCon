#!/usr/bin/env python3
"""
Camera calibration program - Read camera, calculate intrinsic and extrinsic parameters and save them
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
    """Create Charuco calibration board"""
    charuco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    board = cv2.aruco.CharucoBoard((8, 5), 0.031, 0.023, charuco_dict)
    return charuco_dict, board


def estimate_pose(image, charuco_dict, intrinsics_matrix, dist_coeffs, board):
    """Estimate camera pose relative to the marker"""
    detector_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(charuco_dict, detector_params)
    aruco_detector.detectMarkers(image)

    charucodetector = cv2.aruco.CharucoDetector(board)
    charuco_corners, charuco_ids, marker_corners, _ = charucodetector.detectBoard(image)

    print(
        f"Detected {len(marker_corners) if marker_corners is not None else 0} markers"
    )

    if charuco_ids is not None and len(charuco_corners) > 3:
        valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            board,
            intrinsics_matrix,
            dist_coeffs,
            None,
            None,
        )

        if valid:
            # Build transformation matrix
            R_target2cam = cv2.Rodrigues(rvec)[0]
            t_target2cam = tvec.reshape(3, 1)
            target2cam = np.eye(4)
            target2cam[:3, :3] = R_target2cam
            target2cam[:3, 3] = t_target2cam.reshape(-1)

            # Apply coordinate system transformation
            local_frame_change_transform = np.zeros((4, 4))
            local_frame_change_transform[1, 0] = 1
            local_frame_change_transform[0, 1] = 1
            local_frame_change_transform[2, 2] = -1
            local_frame_change_transform[3, 3] = 1

            cam_to_marker = local_frame_change_transform @ np.linalg.inv(target2cam)
            return cam_to_marker, rvec, tvec

    return None, None, None


def calibrate_realsense(camera_serial=None, width=640, height=480, fps=30):
    """Calibrate using RealSense camera"""
    if not REALSENSE_AVAILABLE:
        print("Error: pyrealsense2 not installed")
        return None, None, None, None

    try:
        # Get connected devices
        context = rs.context()
        devices = context.query_devices()

        if len(devices) == 0:
            print("Error: No RealSense camera detected")
            return None, None, None, None

        # Select camera
        device = None
        if camera_serial:
            # Use specified serial number
            for dev in devices:
                if dev.get_info(rs.camera_info.serial_number) == camera_serial:
                    device = dev
                    break
            if device is None:
                print(f"Error: Camera with serial number {camera_serial} not found")
                print("Available camera serial numbers:")
                for dev in devices:
                    print(f"  - {dev.get_info(rs.camera_info.serial_number)}")
                return None, None, None, None
        else:
            # Use the first available camera
            device = devices[0]
            camera_serial = device.get_info(rs.camera_info.serial_number)

        print(f"Using camera: {device.get_info(rs.camera_info.name)}")
        print(f"Serial number: {camera_serial}")

        # Initialize camera
        pipeline = rs.pipeline()
        config = rs.config()

        # Enable specified device
        config.enable_device(camera_serial)

        # Configure stream
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

        # Start stream
        profile = pipeline.start(config)

        # Get camera intrinsics
        color_profile = profile.get_stream(rs.stream.color)
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

        # Build intrinsics matrix
        intrinsics_matrix = np.array(
            [
                [color_intrinsics.fx, 0, color_intrinsics.ppx],
                [0, color_intrinsics.fy, color_intrinsics.ppy],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        # Distortion coefficients
        dist_coeffs = np.array(
            [
                color_intrinsics.coeffs[0],  # k1
                color_intrinsics.coeffs[1],  # k2
                color_intrinsics.coeffs[2],  # p1
                color_intrinsics.coeffs[3],  # p2
                color_intrinsics.coeffs[4],  # k3
            ],
            dtype=np.float64,
        )

        print(f"Camera initialization complete: {width}x{height}@{fps}fps")
        print("Warming up camera...")

        # Warm up for 1 second
        start_time = time.time()
        while time.time() - start_time < 1.0:
            pipeline.wait_for_frames()

        print("Starting marker detection...")

        # Create Charuco calibration board
        charuco_dict, board = create_charuco_board()

        # Get multiple frames for detection
        cam_to_marker = None
        attempts = 0
        max_attempts = 30

        while cam_to_marker is None and attempts < max_attempts:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()

            if color_frame:
                # Convert to numpy array
                color_image = np.asanyarray(color_frame.get_data())
                # RGB to BGR for OpenCV
                bgr_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)

                # Estimate pose
                cam_to_marker, _, _ = estimate_pose(
                    bgr_image, charuco_dict, intrinsics_matrix, dist_coeffs, board
                )

                if cam_to_marker is not None:
                    print("Successfully detected marker!")
                    break

            attempts += 1
            time.sleep(0.1)

        # Stop camera
        pipeline.stop()

        if cam_to_marker is None:
            print(f"Warning: Failed to detect marker (tried {attempts} times)")
            return intrinsics_matrix, dist_coeffs, None, camera_serial

        return intrinsics_matrix, dist_coeffs, cam_to_marker, camera_serial

    except Exception as e:
        print(f"Camera calibration error: {e}")
        return None, None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Camera calibration - Get intrinsic and extrinsic parameters"
    )
    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="Camera serial number (optional, if not specified, use the first available camera)",
    )
    parser.add_argument(
        "--list", action="store_true", help="List all available cameras"
    )
    parser.add_argument("--width", type=int, default=640, help="Camera width")
    parser.add_argument("--height", type=int, default=480, help="Camera height")
    parser.add_argument("--fps", type=int, default=30, help="Camera FPS")

    args = parser.parse_args()

    # If requested to list cameras
    if args.list:
        if not REALSENSE_AVAILABLE:
            print("Error: pyrealsense2 not installed")
            return

        context = rs.context()
        devices = context.query_devices()

        if len(devices) == 0:
            print("No RealSense camera detected")
        else:
            print("Available RealSense cameras:")
            for i, dev in enumerate(devices):
                print(f"  {i+1}. {dev.get_info(rs.camera_info.name)}")
                print(
                    f"     Serial number: {dev.get_info(rs.camera_info.serial_number)}"
                )
                if dev.get_info(rs.camera_info.firmware_version):
                    print(
                        f"     Firmware version: {dev.get_info(rs.camera_info.firmware_version)}"
                    )
        return

    print("Starting camera calibration...")

    # Execute calibration
    intrinsics, dist_coeffs, extrinsics, camera_serial = calibrate_realsense(
        args.serial, args.width, args.height, args.fps
    )

    if intrinsics is None:
        print("Calibration failed")
        return

    # Prepare data to save
    calibration_data = {
        "camera_serial": camera_serial,
        "intrinsics": intrinsics,
        "dist_coeffs": dist_coeffs,
        "extrinsics": extrinsics,  # cam_to_marker transformation
        "image_size": (args.width, args.height),
        "timestamp": time.time(),
    }

    # Save file - named using camera serial number
    output_file = f"{camera_serial}.npy"
    np.save(output_file, calibration_data, allow_pickle=True)
    print(f"\nCalibration data saved to: {output_file}")

    # Print results
    print("\n==================== Calibration Results ====================")
    print(f"Camera serial number: {camera_serial}")
    print(f"Image size: {args.width}x{args.height}")

    print("\nIntrinsics matrix:")
    print(intrinsics)

    print("\nDistortion coefficients:")
    print(dist_coeffs)

    if extrinsics is not None:
        print("\nExtrinsics matrix (camera to marker transformation):")
        print(extrinsics)

        # Extract translation and rotation
        translation = extrinsics[:3, 3]
        from scipy.spatial.transform import Rotation as R

        rotation = R.from_matrix(extrinsics[:3, :3])
        euler_angles = rotation.as_euler("xyz", degrees=True)

        print(f"\nTranslation (m):")
        print(f"  X: {translation[0]:.4f}")
        print(f"  Y: {translation[1]:.4f}")
        print(f"  Z: {translation[2]:.4f}")

        print(f"\nRotation (degrees):")
        print(f"  Roll:  {euler_angles[0]:.2f}")
        print(f"  Pitch: {euler_angles[1]:.2f}")
        print(f"  Yaw:   {euler_angles[2]:.2f}")
    else:
        print("\nExtrinsics: Marker not detected, cannot calculate extrinsics")

    print("=" * 50)


if __name__ == "__main__":
    main()
