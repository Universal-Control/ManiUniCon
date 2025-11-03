from typing import List

from PIL import Image
import torch
import numpy as np


def resize_image_sequence(images, target_size, resample=Image.BICUBIC):
    """
    Resize an image sequence using PIL to match training data processing when converting from zarr to rlds

    Args:
        images: numpy array of shape (N, H, W, C) where:
               N = number of images
               H = height
               W = width
               C = channels
        target_size: int or tuple of (height, width)
                    if int, both height and width will be set to this value
        resample: PIL resampling filter (default: Image.BICUBIC)

    Returns:
        resized images array of shape (N, new_size, new_size, 3)
        Note: output will always have 3 channels (RGB)
    """
    # Convert target_size to tuple if it's a single integer
    if isinstance(target_size, int):
        target_size = (target_size, target_size)
    new_H, new_W = target_size

    N, H, W, C = images.shape

    # Preallocate output array (always 3 channels for RGB)
    output = np.empty((N, new_H, new_W, 3), dtype=np.uint8)

    # Resize each image using PIL
    for i in range(N):
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(images[i])

        # Convert to RGB (regardless of original number of channels)
        img_pil = img_pil.convert("RGB")

        # Resize image
        img_resized = img_pil.resize((new_W, new_H), resample=resample)

        # Convert back to numpy array and store in output
        output[i] = np.array(img_resized)

    return output


class SpatialVLAImageWrapper:
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
