from typing import List

import cv2
import torch
import numpy as np


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


class RoboVlmsImageWrapper:
    def __init__(
        self,
        camera_config,
        state_keys: List[str],
        num_cams: int = 1,
        shared_storage=None,
        device: str = "cpu",
        observation_horizon: int = 1,
    ):
        self.camera_config = camera_config
        self.state_keys = state_keys
        self.num_cams = num_cams
        self.shared_storage = shared_storage
        self.device = device

    def __call__(self, state, camera):
        """
        Wraps the observation for the RoboVlms model.
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

        obs_tensor = {
            "state": state_tensor,
            "image": images_tensor,
        }
        return obs_tensor
