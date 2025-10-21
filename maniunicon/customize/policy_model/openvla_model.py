from typing import Optional, List, Dict, Any, Union
import os
import numpy as np
from collections import deque
from PIL import Image
import torch
import cv2 as cv
import time
from pathlib import Path
from dataclasses import dataclass

# Import OpenVLA components from the installed package
from experiments.robot.openvla_utils import (
    get_vla,
    get_processor,
    get_action_head,
    get_proprio_projector,
    get_vla_action,
    prepare_images_for_vla,
    DEVICE,
)
from experiments.robot.robot_utils import get_image_resize_size
from prismatic.vla.constants import ACTION_DIM, PROPRIO_DIM


class ActionEnsembler:
    def __init__(self, pred_action_horizon, action_ensemble_temp=0.0):
        self.pred_action_horizon = pred_action_horizon
        self.action_ensemble_temp = action_ensemble_temp
        self.action_history = deque(maxlen=self.pred_action_horizon)

    def reset(self):
        self.action_history.clear()

    def ensemble_action(self, cur_action):
        self.action_history.append(cur_action)
        num_actions = len(self.action_history)
        if cur_action.ndim == 1:
            curr_act_preds = np.stack(self.action_history)
        else:
            curr_act_preds = np.stack(
                [
                    pred_actions[i]
                    for (i, pred_actions) in zip(
                        range(num_actions - 1, -1, -1), self.action_history
                    )
                ]
            )
        # if temp > 0, more recent predictions get exponentially *less* weight than older predictions
        weights = np.exp(-self.action_ensemble_temp * np.arange(num_actions))
        weights = weights / weights.sum()
        # compute the weighted average across all predictions for this timestep
        cur_action = np.sum(weights[:, None] * curr_act_preds, axis=0)

        return cur_action


@dataclass
class OpenVLAConfig:
    """Configuration for OpenVLA model wrapper"""

    # Model configuration
    pretrained_checkpoint: Union[str, Path] = "openvla/openvla-7b"
    model_family: str = "openvla"

    # Action head configuration
    use_l1_regression: bool = False
    use_diffusion: bool = False
    num_diffusion_steps_train: int = 50
    num_diffusion_steps_inference: int = 50

    # Input configuration
    use_film: bool = False
    num_images_in_input: int = 1
    use_proprio: bool = False
    center_crop: bool = False

    # LoRA configuration
    lora_rank: int = 32

    # Normalization
    unnorm_key: str = ""
    use_relative_actions: bool = False

    # Quantization
    load_in_8bit: bool = False
    load_in_4bit: bool = False

    # Other
    seed: int = 7


class OpenVLAModel:
    def __init__(
        self,
        device,
        saved_model_path: str = "openvla/openvla-7b",
        unnorm_key: str = None,
        image_size: List[int] = [224, 224],
        action_ensemble_temp: float = 0.0,
        use_act_chunk: bool = True,
        task_name: str = None,
        use_l1_regression: bool = False,
        use_diffusion: bool = False,
        num_diffusion_steps_inference: int = 50,
        use_proprio: bool = False,
        center_crop: bool = False,
        num_images_in_input: int = 1,
        lora_rank: int = 32,
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.device = device
        self.unnorm_key = unnorm_key
        print(f"*** unnorm_key: {unnorm_key} ***")

        # Setup task instruction
        self.task_name = task_name

        # Create configuration object
        self.cfg = OpenVLAConfig(
            pretrained_checkpoint=saved_model_path,
            unnorm_key=unnorm_key,
            use_l1_regression=use_l1_regression,
            use_diffusion=use_diffusion,
            num_diffusion_steps_inference=num_diffusion_steps_inference,
            use_proprio=use_proprio,
            center_crop=center_crop,
            num_images_in_input=num_images_in_input,
            lora_rank=lora_rank,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
        )

        # Load VLA model
        print("Loading OpenVLA model...")
        self.vla = get_vla(self.cfg)

        # Load processor
        self.processor = get_processor(self.cfg)

        # Load proprioception projector if needed
        self.proprio_projector = None
        if self.cfg.use_proprio:
            self.proprio_projector = get_proprio_projector(
                self.cfg, self.vla.llm_dim, PROPRIO_DIM
            )

        # Load action head if needed
        self.action_head = None
        if self.cfg.use_l1_regression or self.cfg.use_diffusion:
            self.action_head = get_action_head(self.cfg, self.vla.llm_dim)

        # Image configuration
        self.image_size = image_size
        self.resize_size = get_image_resize_size(self.cfg)

        # Get observation horizon from processor if available
        if hasattr(self.processor, "num_obs_steps") and hasattr(
            self.processor, "obs_delta"
        ):
            self.obs_horizon = (
                self.processor.num_obs_steps - 1
            ) * self.processor.obs_delta + 1
            self.obs_interval = self.processor.obs_delta
        else:
            # Default values if not available
            self.obs_horizon = 1
            self.obs_interval = 1

        # Action chunking configuration
        if hasattr(self.processor, "action_chunk_size"):
            self.pred_action_horizon = self.processor.action_chunk_size
        else:
            # Default based on constants
            from prismatic.vla.constants import NUM_ACTIONS_CHUNK

            self.pred_action_horizon = NUM_ACTIONS_CHUNK

        self.image_history = deque(maxlen=self.obs_horizon)
        self.use_act_chunk = use_act_chunk

        # Action ensemble configuration
        if self.use_act_chunk:
            action_ensemble = False
        else:
            action_ensemble = True
        self.action_ensemble = action_ensemble
        self.action_ensemble_temp = action_ensemble_temp

        if self.action_ensemble:
            self.action_ensembler = ActionEnsembler(
                self.pred_action_horizon, self.action_ensemble_temp
            )
        else:
            self.action_ensembler = None

        self.rollout_step_counter = 0

    def reset(self) -> None:
        print("reset model now!!!!!!!!")
        self.image_history.clear()
        if self.action_ensemble:
            self.action_ensembler.reset()
        self.rollout_step_counter = 0

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        image = cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)
        return image

    def _add_image_to_history(self, image: np.ndarray) -> None:
        if len(self.image_history) == 0:
            self.image_history.extend([image] * self.obs_horizon)
        else:
            self.image_history.append(image)

    def _obtain_image_history(self) -> List[Image.Image]:
        image_history = list(self.image_history)
        images = image_history[:: self.obs_interval]
        images = [Image.fromarray(image).convert("RGB") for image in images]
        return images

    def preprocess(self, obs):
        """Preprocess observation from ManiUniCon format to OpenVLA format"""
        # Extract primary camera image
        obs["image"]["camera_1"] = (
            obs["image"]["camera_1"].squeeze().cpu().detach()
        )  # (224, 224, 3)

        # Process primary image
        image = obs["image"]["camera_1"].numpy()
        assert image.dtype == np.uint8
        image = self._resize_image(image)

        # For OpenVLA, we'll return the image directly for processing
        # The history handling depends on the specific OpenVLA configuration
        if self.obs_horizon > 1:
            self._add_image_to_history(image)
            images = self._obtain_image_history()
        else:
            images = [Image.fromarray(image).convert("RGB")]

        # Prepare observation dict for OpenVLA
        obs_dict = {
            "full_image": image,  # Primary camera as numpy array
        }

        # Add wrist cameras if configured for multi-image input
        if self.cfg.num_images_in_input > 1:
            # Check if there are additional camera views in the observation
            for key in obs["image"].keys():
                if "wrist" in key.lower() or "camera_0" in key:
                    wrist_img = obs["image"][key].squeeze().cpu().detach().numpy()
                    wrist_img = self._resize_image(wrist_img)
                    obs_dict[key] = wrist_img

        # Add proprioception state if available and configured
        if self.cfg.use_proprio and "state" in obs:
            if isinstance(obs["state"], torch.Tensor):
                obs_dict["state"] = obs["state"].cpu().numpy()
            else:
                obs_dict["state"] = obs["state"]

        # Add task instruction
        obs_dict["instruction"] = (
            self.task_name if self.task_name else "complete the task"
        )

        return obs_dict

    def __call__(self, obs, **kwargs):
        """
        Forward pass through the model.
        :param obs: Observation input to the model.
        :return: Model output.
        """
        # Preprocess observation to OpenVLA format
        obs_dict = self.preprocess(obs)

        # Get action from OpenVLA
        start = time.time()
        action_list = get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            obs_dict,
            obs_dict["instruction"],
            action_head=self.action_head,
            proprio_projector=self.proprio_projector,
            use_film=self.cfg.use_film,
        )
        print(f"**** OpenVLA-OFT inference time: {time.time() - start}")

        # Convert list of actions to numpy array
        raw_actions = np.array(action_list)

        # Handle action ensemble if configured
        if self.action_ensemble:
            print("ensemble action!!!")
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)
            if raw_actions.ndim == 1:
                raw_actions = raw_actions[None]

        self.rollout_step_counter += 1

        # Ensure we have numpy array
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.cpu().numpy()

        # Convert rotation from euler to quaternion
        # OpenVLA outputs: [x, y, z, rx, ry, rz, gripper]
        # Need to convert to: [x, y, z, qx, qy, qz, qw, gripper]
        from maniunicon.utils.vla_utils import euler_pose_to_quat
        N, A = raw_actions.shape
        if A == 7:  # Euler format
            raw_actions = np.concatenate(
                [
                    euler_pose_to_quat(
                        raw_actions[..., :-1].reshape(-1, A - 1)
                    ).reshape(N, -1),
                    raw_actions[..., -1:],
                ],
                axis=-1,
            )

        print(f"step {self.rollout_step_counter} action {raw_actions}")

        return raw_actions
