import torch
import torch.nn.functional as F
import numpy as np
import cv2
import time
from pathlib import Path

from maniunicon.utils.misc import dict_apply, convert_to_torch
from maniunicon.utils.pcd_utils import (
    add_gaussian_noise,
    randomly_drop_point,
    sample_pcd_data,
)
from maniunicon.utils.pcd_utils import (
    create_pointcloud_from_depth_batch,
    create_pointcloud_from_rgbd_batch,
    BOUND,
    pcd_filter_bound_torch,
    uniform_sampling_torch,
    capture_pcd,
    capture_depth,
)
from maniunicon.utils.math_utils import quat_from_matrix
from scipy.spatial.transform import Rotation


def resize_image_sequence(images, target_size, interp=cv2.INTER_AREA):
    """
    Resize an image sequence using OpenCV

    Args:
        images: numpy array of shape (N, H, W, C) where:
               N = number of images
               H = height
               W = width
               C = channels
        target_size: tuple of (height, width)

    Returns:
        resized images array of shape (N, new_H, new_W, C)
    """
    N, H, W, C = images.shape
    new_H, new_W = target_size

    # Reshape to 2D array of images for faster processing
    reshaped = images.reshape(-1, H, W, C)

    # Preallocate output array
    output = np.empty((N, new_H, new_W, C), dtype=images.dtype)

    # Resize each image
    for i in range(N):
        res = cv2.resize(images[i], (new_W, new_H), interpolation=interp)
        if C == 1:
            output[i] = res[:, :, np.newaxis]
        else:
            output[i] = res

    return output


class FalconPCDWrapper:
    def __init__(
        self,
        camera_config,
        state_keys,
        pcd_num_points=2048,
        num_cams=1,
        shared_storage=None,
        device="cpu",
        observation_horizon=1,
    ):
        self.camera_config = camera_config
        self.state_keys = state_keys
        self.num_cams = num_cams
        self.pcd_num_points = pcd_num_points
        self.shared_storage = shared_storage
        self.device = device

    def __call__(self, state, camera):
        """
        Wraps the observation for the Falcon-PCD model.
        :return: Wrapped observation tensor.
        """
        assert state is not None, "State cannot be None"
        assert camera is not None, "Camera cannot be None"

        state_tensor = (
            torch.from_numpy(
                np.concatenate(
                    [getattr(state, key) for key in self.state_keys], axis=-1
                )
            )
            .to(self.device)
            .float()
        )

        colors = resize_image_sequence(
            camera.colors.reshape(-1, *camera.colors.shape[2:]), (224, 224)
        )
        colors = colors.reshape(
            camera.colors.shape[0],
            camera.colors.shape[1],
            *colors.shape[1:],
        )

        images_tensor = {}
        for cam_idx in range(colors.shape[1]):
            images_tensor[f"camera_{cam_idx}"] = torch.from_numpy(
                colors[:, cam_idx]
            ).to(self.device)

        pcd_tensor = self.create_pcds_from_camera(camera, self.device)

        obs_tensor = {
            "state": state_tensor,
            "image": images_tensor,
            "pointcloud": pcd_tensor,
        }

        return obs_tensor

    def create_pcds_from_camera(self, camera, device):
        """
        Create point clouds from camera data.
        :param camera: Camera object containing RGB, depth, intrinsics and extrinsics.
        :return: Processed point clouds.
        """
        CAMERA_IDX = 1

        transforms = camera.transforms.reshape(-1, 4, 4)[CAMERA_IDX]
        position = transforms[:3, 3]  # (3)
        # Extract rotation matrix and convert to quaternion (w, x, y, z)
        rotation_matrix = transforms[:3, :3]
        quat_xyzw = Rotation.from_matrix(rotation_matrix).as_quat()
        # Convert to (w, x, y, z) format for create_pointcloud_from_depth_batch
        orientation = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        position = convert_to_torch(position, dtype=torch.float32, device=device)
        orientation = convert_to_torch(orientation, dtype=torch.float32, device=device)

        intrinsics = convert_to_torch(
            camera.intrs.reshape(-1, 4), dtype=torch.float32, device=device
        )  # n_cams, 4
        intrinsics = intrinsics[CAMERA_IDX]
        # Extract fx, fy, cx, cy from intrinsics
        fx, fy, cx, cy = intrinsics[0], intrinsics[1], intrinsics[2], intrinsics[3]
        # Build (3,3) intrinsic matrices for camera_1
        intrinsic_matrix = torch.zeros(3, 3)
        intrinsic_matrix[0, 0] = fx
        intrinsic_matrix[1, 1] = fy
        intrinsic_matrix[0, 2] = cx
        intrinsic_matrix[1, 2] = cy
        intrinsic_matrix[2, 2] = 1.0

        depths = convert_to_torch(
            camera.depths.reshape(-1, camera.depths.shape[2], camera.depths.shape[3]),
            dtype=torch.float32,
            device=device,
        )  # n_cams, H, W
        depth = depths[CAMERA_IDX]

        pcd_tensor = create_pointcloud_from_depth_batch(
            intrinsic_matrix=intrinsic_matrix,
            depth=depth,
            keep_invalid=False,
            position=position,
            orientation=orientation,
            device=device,
        )

        return pcd_tensor
