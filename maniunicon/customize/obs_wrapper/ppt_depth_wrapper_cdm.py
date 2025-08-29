import torch
import numpy as np
import cv2
import time
from collections import deque

from maniunicon.utils.misc import dict_apply, temporal_depth_proc_batch
from maniunicon.utils.timestamp_accumulator import TimestampAlignedBuffer

try:
    from ranging_anything.compute_metric import (
        interp_depth_rgb,
        recover_metric_depth_ransac,
    )
except ImportError:
    print("ranging_anything not found, depth model will not be available.")

EPS = 1e-3


class PPTDepthWrapper:
    def __init__(
        self,
        camera_config,
        state_keys,
        num_cams=1,
        depth_model=None,
        depth_size=None,
        tar_size=None,
        depth_clip=None,
        shared_storage=None,
        device="cpu",
        temporal_ensemble=False,
        temporal_stack=False,
    ):
        self.camera_config = camera_config
        self.camera_names = list(camera_config.camera_names.keys())
        self.state_keys = state_keys
        self.num_cams = num_cams

        self.depth_size = None
        self.tar_size = tar_size
        self.depth_model = None
        self.shared_storage = shared_storage
        if depth_clip is not None:
            assert (
                isinstance(depth_clip, (list, tuple)) and len(depth_clip) == 2
            ), "depth_clip must be a list or tuple of two values"
        self.depth_clip = depth_clip
        self.record_buffer = None
        if depth_model is not None:
            assert (
                depth_size is not None
            ), "depth_size must be provided if depth_model is used"

            self.depth_model = DepthModelWrapper(depth_model)
            self.depth_size = depth_size

            self.depth_model.eval()
            self.depth_model.to(device)
            print("Depth model applied to camera data.")

            self.temporal_ensemble = temporal_ensemble
            self.temporal_stack = temporal_stack
            self.prev = None
            self.prev_depths = deque(maxlen=2)
            if temporal_ensemble:
                print("Using temporal ensemble for depth model.")
            if temporal_stack:
                print("Using temporal stacking for depth model.")

            # example_depth = np.random.randn(
            #     1, self.depth_size[0], self.depth_size[1], 1
            # ).astype(np.float32)
            # example_color = np.ones(
            #     (1, self.depth_size[0], self.depth_size[1], 3), dtype=np.uint8
            # )
            # # Warm-up runs
            # for _ in range(5):
            #     _ = self.depth_model(example_color, example_depth)

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

        if self.depth_model is not None:
            colors = camera.colors
            colors = colors.reshape(-1, *colors.shape[2:]).astype(
                np.float32
            )  # horizon * n_cams, H, W, 3
            depths = camera.depths.reshape(
                -1, *camera.depths.shape[2:]
            )  # horizon * n_cams, H, W

            # resized_colors = []
            # resized_depths = []
            # for i in range(len(depths)):
            #     color = cv2.resize(
            #         colors[i], self.tar_size[::-1], interpolation=cv2.INTER_AREA
            #     )
            #     depth = cv2.resize(
            #         depths[i], self.tar_size[::-1], interpolation=cv2.INTER_NEAREST
            #     )
            #     resized_colors.append(color)
            #     resized_depths.append(depth)
            # colors = np.stack(resized_colors, axis=0)
            # depths = np.stack(resized_depths, axis=0

            torch.cuda.synchronize() if self.device == "cuda" else None
            time1 = time.time()

            if self.temporal_stack:
                if len(self.prev_depths) < 2:
                    self.prev_depths.append(depths)
                    self.prev_depths.append(depths)

                input_depths = np.stack(list(self.prev_depths) + [depths], axis=-1)
                self.prev_depths.append(depths)
            else:
                input_depths = depths

            with torch.no_grad():
                pred_depths = self.depth_model(colors, input_depths)
            torch.cuda.synchronize() if self.device == "cuda" else None
            print("infer used time", time.time() - time1)

            # capture_depth(pred_depths, "pred_depth")
            pred_depths = pred_depths.reshape(-1, self.num_cams, *pred_depths.shape[1:])
            if self.temporal_ensemble:
                if self.prev is None:
                    self.prev = (colors, pred_depths)
                    cur = (colors, pred_depths)
                    pred_depths = temporal_depth_proc_batch(cur, self.prev)
                    self.prev = cur

            camera.depths = pred_depths

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

        # Convert from [x,y,z,w] to [w,x,y,z] and ensure w is positive
        state.tcp_orientation = np.concatenate(
            [state.tcp_orientation[:, 3:], state.tcp_orientation[:, :3]], axis=-1
        )
        state.tcp_orientation[
            state.tcp_orientation[:, 0] < 0
        ] *= -1  # Flip quaternion if w is negative

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


class DepthModelWrapper(torch.nn.Module):
    def __init__(self, depth_model):
        super().__init__()
        for para in depth_model.parameters():
            para.requires_grad = False
        self.depth_model = depth_model

    def forward(self, x, lowres_depth):
        # for batch_idx in range(x.shape[0]): # Only consider 1 image for now

        lowres_depth[lowres_depth > 0] = 1 / lowres_depth[lowres_depth > 0]

        depth = self.depth_model.infer_image(
            x.squeeze(),
            lowres_depth.squeeze(),
            input_size=518,
        )[None, ...]

        # depth[depth > 0] = 1. / depth[depth > 0]  # Convert to inverse depth
        # depth[depth <= 0] = 100000000  # Convert to inverse depth

        # depth_normalized = cv2.normalize(depth[0], None, 0, 255, cv2.NORM_MINMAX)
        # depth_colormap = cv2.applyColorMap(
        #     depth_normalized.astype(np.uint8), cv2.COLORMAP_JET
        # )
        # cv2.imshow("depth", depth_colormap)
        # cv2.waitKey(1)
        # depth = np.clip(depth, 0.01, 6.0)  # Clip depth values
        return 1.0 / depth


def save_data(depth_image, color_image, camera_dir, frame_count):
    depth_image = depth_image * 1000
    depth_image = np.nan_to_num(depth_image, 0)
    depth_image[depth_image > 65535] = 65535
    depth_image[depth_image < 1e-5] = 0

    cv2.imwrite(
        str(camera_dir / f"depth_{frame_count}.png"), depth_image.astype(np.uint16)
    )
    cv2.imwrite(str(camera_dir / f"color_{frame_count}.png"), color_image)
