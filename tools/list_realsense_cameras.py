import pyrealsense2 as rs


def list_realsense_cameras():
    """List all connected RealSense cameras with their details and intrinsic parameters."""
    ctx = rs.context()
    devices = ctx.query_devices()

    if len(list(devices)) == 0:
        print("No RealSense cameras found!")
        return

    print("\nConnected RealSense Cameras:")
    print("-" * 50)

    for i, device in enumerate(devices):
        print(f"\nCamera {i}:")
        print(f"  Serial Number: {device.get_info(rs.camera_info.serial_number)}")
        print(f"  Name: {device.get_info(rs.camera_info.name)}")
        print(f"  USB Type: {device.get_info(rs.camera_info.usb_type_descriptor)}")
        print(f"  Firmware Version: {device.get_info(rs.camera_info.firmware_version)}")

        # Get supported resolutions and intrinsics
        print("\n  Supported Resolutions and Intrinsics:")
        for sensor in device.sensors:
            if sensor.supports(rs.camera_info.product_id):
                for profile in sensor.get_stream_profiles():
                    if profile.stream_type() == rs.stream.color:
                        vprofile = profile.as_video_stream_profile()
                        print(
                            f"    Color: {vprofile.width()}x{vprofile.height()} @ {vprofile.fps()}fps"
                        )

                        # Get intrinsics for color stream
                        try:
                            intrinsics = vprofile.get_intrinsics()
                            print("      Intrinsics:")
                            print(
                                f"        Focal Length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}"
                            )
                            print(
                                f"        Principal Point: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}"
                            )
                            print(f"        Distortion Model: {intrinsics.model}")
                            print(
                                f"        Distortion Coefficients: {intrinsics.coeffs}"
                            )
                        except Exception as e:
                            print(f"      Could not get intrinsics: {e}")

                    elif profile.stream_type() == rs.stream.depth:
                        vprofile = profile.as_video_stream_profile()
                        print(
                            f"    Depth: {vprofile.width()}x{vprofile.height()} @ {vprofile.fps()}fps"
                        )

                        # Get intrinsics for depth stream
                        try:
                            intrinsics = vprofile.get_intrinsics()
                            print("      Intrinsics:")
                            print(
                                f"        Focal Length: fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}"
                            )
                            print(
                                f"        Principal Point: cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}"
                            )
                            print(f"        Distortion Model: {intrinsics.model}")
                            print(
                                f"        Distortion Coefficients: {intrinsics.coeffs}"
                            )
                        except Exception as e:
                            print(f"      Could not get intrinsics: {e}")

        print("-" * 50)


def get_active_camera_intrinsics():
    """Get intrinsics from an active camera stream."""
    print("\nStarting camera to get active intrinsics...")

    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color and depth streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    try:
        # Start streaming
        profile = pipeline.start(config)

        # Get stream profiles
        color_profile = profile.get_stream(rs.stream.color)
        depth_profile = profile.get_stream(rs.stream.depth)

        # Get intrinsics
        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        print("\nActive Camera Intrinsics:")
        print("=" * 50)

        print("\nColor Stream Intrinsics:")
        print(f"  Resolution: {color_intrinsics.width} x {color_intrinsics.height}")
        print(
            f"  Focal Length: fx={color_intrinsics.fx:.2f}, fy={color_intrinsics.fy:.2f}"
        )
        print(
            f"  Principal Point: cx={color_intrinsics.ppx:.2f}, cy={color_intrinsics.ppy:.2f}"
        )
        print(f"  Distortion Model: {color_intrinsics.model}")
        print(f"  Distortion Coefficients: {color_intrinsics.coeffs}")

        print("\nDepth Stream Intrinsics:")
        print(f"  Resolution: {depth_intrinsics.width} x {depth_intrinsics.height}")
        print(
            f"  Focal Length: fx={depth_intrinsics.fx:.2f}, fy={depth_intrinsics.fy:.2f}"
        )
        print(
            f"  Principal Point: cx={depth_intrinsics.ppx:.2f}, cy={depth_intrinsics.ppy:.2f}"
        )
        print(f"  Distortion Model: {depth_intrinsics.model}")
        print(f"  Distortion Coefficients: {depth_intrinsics.coeffs}")

        # Get extrinsics (transformation between color and depth)
        try:
            extrinsics = depth_profile.get_extrinsics_to(color_profile)
            print("\nExtrinsics (Depth to Color):")
            print("  Rotation Matrix:")
            for i in range(3):
                print(f"    {extrinsics.rotation[i*3:(i+1)*3]}")
            print(f"  Translation Vector: {extrinsics.translation}")
        except Exception as e:
            print(f"  Could not get extrinsics: {e}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()


if __name__ == "__main__":
    list_realsense_cameras()
    print("\n" + "=" * 70)
    get_active_camera_intrinsics()
