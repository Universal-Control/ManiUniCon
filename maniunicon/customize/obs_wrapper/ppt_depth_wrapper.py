import torch
import numpy as np
import cv2

from maniunicon.utils.misc import dict_apply
from maniunicon.utils.timestamp_accumulator import TimestampAlignedBuffer

EPS = 1e-3


class PPTDepthWrapper:
    def __init__(
        self,
        camera_config,
        state_keys,
        num_cams=1,
        tar_size=None,
        depth_clip=None,
        shared_storage=None,
        device="cpu",
    ):
        self.camera_config = camera_config
        self.camera_names = list(camera_config.camera_names.keys())
        self.state_keys = state_keys
        self.num_cams = num_cams
        self.tar_size = tar_size
        self.shared_storage = shared_storage
        if depth_clip is not None:
            assert (
                isinstance(depth_clip, (list, tuple)) and len(depth_clip) == 2
            ), "depth_clip must be a list or tuple of two values"
        self.depth_clip = depth_clip
        self.record_buffer = None
        self.device = device

    def _dump_data(self):
        record_dir = self.shared_storage.get_record_dir()
        if not record_dir:
            print("Record directory not set, skipping data dump")
            return

        self.record_buffer.dump(name="realsense_model", dir=record_dir)

    def __call__(self, state, camera):
        """
        Wraps the observation for the PPT model.
        :return: Wrapped observation tensor.
        """
        assert state is not None, "State cannot be None"
        assert camera is not None, "Camera cannot be None"

        if self.shared_storage.is_recording.value:
            if self.record_buffer is None:
                self.record_buffer = TimestampAlignedBuffer(
                    self.shared_storage.record_start_time.value,
                    self.shared_storage.record_dt.value,
                    self.shared_storage.max_record_steps,
                    overwrite=False,
                )
            self.record_buffer.add(
                {k: v[0] for k, v in camera.model_dump().items()},
                timestamp=camera.timestamp.item(),
            )
        else:
            if self.record_buffer is not None:
                self._dump_data()
                self.record_buffer = None

        depths = {}
        for idx in range(self.num_cams):
            depth = camera.depths[:, idx][..., np.newaxis]  # horizon, H, W, 1
            if self.depth_clip is not None:
                depth = np.clip(depth, self.depth_clip[0], self.depth_clip[1])
            depth = resize_image_sequence(depth, self.tar_size, cv2.INTER_NEAREST)
            depths[self.camera_names[idx]] = (
                torch.from_numpy(depth).float().to(self.device)
            )

        obs = {
            "depth": depths,
            "state": torch.from_numpy(
                np.concatenate(
                    [getattr(state, key) for key in self.state_keys], axis=-1
                )
            )
            .float()
            .to(self.device),
        }
        obs = preprocess_obs(obs)

        obs_tensor = dict_apply(
            obs,
            lambda x: x.flatten(0, 1),
        )
        # general_capture(obs_tensor, visualize=True)

        return obs_tensor


def preprocess_obs(sample):
    def unsqueeze(x, dim=0):
        if isinstance(x, np.ndarray):
            return np.expand_dims(x, dim)
        elif isinstance(x, torch.Tensor):
            return x.unsqueeze(dim)
        else:
            raise ValueError(f"Unsupported type: {type(x)}")

    sample = dict_apply(sample, unsqueeze)

    return sample


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


def save_data(depth_image, color_image, camera_dir, frame_count):
    depth_image = depth_image * 1000
    depth_image = np.nan_to_num(depth_image, 0)
    depth_image[depth_image > 65535] = 65535
    depth_image[depth_image < 1e-5] = 0

    cv2.imwrite(
        str(camera_dir / f"depth_{frame_count}.png"), depth_image.astype(np.uint16)
    )
    cv2.imwrite(str(camera_dir / f"color_{frame_count}.png"), color_image)
