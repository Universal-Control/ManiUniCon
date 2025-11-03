from typing import Optional, Sequence, List
import os
import numpy as np
from transformers import AutoModel, AutoProcessor
from collections import deque
from PIL import Image
import torch
import cv2 as cv
import time


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


class SpatialVLAModel:
    def __init__(
        self,
        device,
        saved_model_path: str = "IPEC-COMMUNITY/spatialvla-4b-224-pt",
        unnorm_key: str = None,
        image_size: list[int] = [224, 224],
        action_ensemble_temp: float = -0.8,
        use_act_chunk: bool = False,
        task_name: str = None,
    ) -> None:
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        self.device = device
        self.unnorm_key = unnorm_key
        print(f"*** unnorm_key: {unnorm_key} ***")
        # setup the task instruction that input to vla model
        self.task_name = task_name

        self.processor = AutoProcessor.from_pretrained(
            saved_model_path, trust_remote_code=True
        )
        self.vla = AutoModel.from_pretrained(
            saved_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()
        self.vla.to(self.device)

        self.image_size = image_size
        self.obs_horizon = (
            self.processor.num_obs_steps - 1
        ) * self.processor.obs_delta + 1
        self.obs_interval = self.processor.obs_delta
        self.pred_action_horizon = self.processor.action_chunk_size
        self.image_history = deque(maxlen=self.obs_horizon)
        self.use_act_chunk = use_act_chunk

        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

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
        self.sticky_action_is_on = False
        self.gripper_action_repeat = 0
        self.sticky_gripper_action = 0.0
        self.previous_gripper_action = None

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
        obs["image"]["camera_1"] = (
            obs["image"]["camera_1"].squeeze().cpu().detach()
        )  # (224, 224, 3)
        # preprocess image
        image = obs["image"]["camera_1"].numpy()
        assert image.dtype == np.uint8
        image = self._resize_image(image)
        self._add_image_to_history(image)
        images: List[Image.Image] = self._obtain_image_history()

        return images

    def __call__(self, obs, **kwargs):
        """
        Forward pass through the model.
        :param obs: Observation input to the model.
        :return: Model output.
        """
        """Step function."""
        images = self.preprocess(obs)
        prompt = self.task_name

        # predict action (7-dof; un-normalize for bridgev2)
        start = time.time()
        inputs = self.processor(
            images=images,
            text=prompt,
            unnorm_key=self.unnorm_key,
            return_tensors="pt",
            do_normalize=False,
        )
        with torch.no_grad():
            if hasattr(self.processor, "action_tokenizer"):
                generation_outputs = self.vla.predict_action(inputs)

                raw_actions = self.processor.decode_actions(
                    generation_outputs=generation_outputs,
                    unnorm_key=self.unnorm_key,
                )["actions"]
            else:
                raw_actions = self.vla.predict_action(**inputs)["actions"]
                raw_actions = raw_actions.cpu().numpy()
        # cal policy infer time
        print("**** SpatialVLA inference time: ", time.time() - start)

        if self.action_ensemble:
            print("ensemble action!!!")
            raw_actions = self.action_ensembler.ensemble_action(raw_actions)[None]

        self.rollout_step_counter += 1
        if isinstance(raw_actions, torch.Tensor):
            raw_actions = raw_actions.numpy()

        # convert rot from euler to quat
        from maniunicon.utils.vla_utils import euler_pose_to_quat

        N, A = raw_actions.shape
        raw_actions = np.concatenate(
            [
                euler_pose_to_quat(raw_actions[..., :-1].reshape(-1, A - 1)).reshape(
                    N, -1
                ),
                raw_actions[..., -1:],
            ],
            axis=-1,
        )
        print(f"step {self.rollout_step_counter} action {raw_actions}")

        return raw_actions
