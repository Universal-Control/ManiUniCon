import json
import os.path
import shutil
from copy import deepcopy
import torch
from PIL import Image
from typing import Literal
import numpy as np
import functools
import time

# Import RoboVLMs components from the installed package
from robovlms.train.base_trainer import BaseTrainer
from robovlms.utils.model_utils import build_tokenizer
from robovlms.data.data_utils import get_text_function
from robovlms.data.data_utils import (
    preprocess_image,
    get_prompt_builder,
    tcp_to_world_frame,
)
from robovlms.utils.config_utils import load_config
from queue import Queue
from robovlms.model.policy_head.action_tokenizer import ActionTokenizer

fwd_decay_ratio = 1


class RoboVlmsModel:
    # model option
    def __init__(
        self,
        ckpt_path,
        config_path,
        device,
        log_save_dir=None,
        raw_calvin=True,
        debug=False,
        use_act_chunk=False,
        task_name=None,
    ):
        # setup the task instruction that input to vla model
        self.task_name = task_name
        # Loading robovlms model configs
        assert config_path != None
        configs = load_config(config_path)
        self.model = BaseTrainer(configs=configs)

        # Get checkpoint path
        print("ckpt_path", ckpt_path)
        from robovlms.utils.zero_to_fp32 import (
            convert_zero_checkpoint_to_fp32_state_dict,
        )

        # Handle DeepSpeed ckpt
        if os.path.isdir(ckpt_path):
            target_ckpt_path = ckpt_path.replace(".ckpt", ".pt")
            print(f"converting {ckpt_path} to {target_ckpt_path}")
            convert_zero_checkpoint_to_fp32_state_dict(ckpt_path, target_ckpt_path)
            ckpt_path = target_ckpt_path

        self.init_config(
            ckpt_path, configs, device, log_save_dir, raw_calvin, debug, use_act_chunk
        )
        # self.model.model.lm_head.window_size = 1

    def init_config(
        self,
        ckpt_path,
        configs,
        device,
        save_dir=None,
        raw_calvin=False,
        debug=False,
        use_act_chunk=False,
    ):
        ### load and convert checkpoint
        self.debug = debug
        self.use_act_chunk = use_act_chunk
        if configs["model"] == "kosmos":
            import transformers

            package_dir = transformers.__path__[0]
            robovlms_root = os.environ.get("ROBOVLMS_ROOT")
            if not robovlms_root:
                raise EnvironmentError(
                    "ROBOVLMS_ROOT environment variable must be set to the RoboVLMs installation directory."
                )
            source_model_path = os.path.join(
                robovlms_root, "tools", "modeling_kosmos2.py"
            )
            target_model_path = os.path.join(
                package_dir, "models", "kosmos2", "modeling_kosmos2.py"
            )
            # Copy the custom kosmos model implementation into the transformers package when needed.
            shutil.copy(source_model_path, target_model_path)

        if not self.debug:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            if "state_dict" in ckpt:
                new_state_dict = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                new_state_dict = ckpt["model_state_dict"]
            else:
                raise KeyError("no checkpoint dict in the loaded pretrain parameters")

            new_state_dict = self.convert_old_state_dict(new_state_dict)
            msg = self.model.load_state_dict(new_state_dict, strict=False)
            print(f"RoboVLMs CKPT Loaded \n {msg}")
            del new_state_dict

            ckpt_dir = os.path.dirname(ckpt_path)
            ckpt_name = os.path.basename(ckpt_path)
            save_dir = ckpt_dir if save_dir is None else save_dir
            load_info_path = os.path.join(save_dir, f"{ckpt_name}_loading_msg.json")
            if os.path.exists(load_info_path):
                os.system(f"rm {load_info_path}")
            with open(load_info_path, "w") as f:
                _info = {
                    "missing_keys": msg.missing_keys,
                    "unexpected_keys": msg.unexpected_keys,
                }
                json.dump(_info, f, indent=2)
                print(f"Model loading msg is updated to: {load_info_path}")

        self.configs = configs

        dtype = torch.float32
        if self.configs["trainer"]["precision"] == "bf16":
            dtype = torch.bfloat16
        elif self.configs["trainer"]["precision"] == "fp16":
            dtype = torch.float16
        self.dtype = dtype
        self.act_head_configs = self.configs["act_head"]
        self.raw_calvin = raw_calvin
        self.tcp_rel = self.configs.get("tcp_rel", False)

        print(f"raw action: {self.raw_calvin}")

        self.device = device
        self.policy = self.model
        self.policy = self.policy.to(self.dtype)
        # self.policy = self.policy.float()
        self.policy.to(self.device)
        self.policy.eval()

        if not hasattr(self.policy.model, "lm_head"):
            self.policy.model.lm_head = self.policy.model.act_head

        self.tokenizer = build_tokenizer(self.configs["tokenizer"])

        self.window_size = configs["window_size"]
        self.fwd_pred_next_n = configs["fwd_pred_next_n"]
        self.act_step = self.fwd_pred_next_n + 1
        self.seq_len = self.configs["seq_len"]
        self.use_hand_rgb = self.configs["use_hand_rgb"]

        if hasattr(self, "policy_setup"):
            data_mix = "bridge" if self.policy_setup == "widowx_bridge" else "rt_1"
            configs["train_dataset"]["data_mix"] = data_mix
            configs["val_dataset"]["data_mix"] = data_mix

        image_preprocess = self.model.model.image_processor
        self.image_preprocess = functools.partial(
            preprocess_image,
            image_processor=image_preprocess,
            model_type=configs["model"],
        )

        self.text_preprocess = get_text_function(
            self.model.model.tokenizer, configs["model"]
        )

        self.action_space = self.configs["act_head"].get("action_space", "continuous")
        if self.action_space == "discrete":
            self.action_tokenizer = ActionTokenizer(
                self.tokenizer,
                bins=self.act_head_configs["n_bin"],
                min_action=self.act_head_configs["min_action"],
                max_action=self.act_head_configs["max_action"],
            )

        print(f"Evaluating checkpoint {ckpt_path}")

        self.rgb_list = []
        self.hand_rgb_list = []
        self.action_hist_list = []
        self.rollout_step_counter = 0

        self.vision_queue = Queue(maxsize=self.window_size)
        self.vision_gripper_queue = Queue(maxsize=self.window_size)
        self.action_queue = Queue(maxsize=self.window_size - 1)

    def ensemble_action(self, action):
        if action.ndim >= 3:
            action = action.squeeze()

        if action.ndim == 1:
            action = action.unsqueeze(0)

        self.action_hist_list.append(action)

        act_cache = []
        max_len = 5
        while len(self.action_hist_list) > max_len:
            self.action_hist_list.pop(0)

        idx = 0
        for act in self.action_hist_list[::-1]:
            act_cache.append(act[idx])
            idx += 1

        act_cache = torch.stack(act_cache, dim=0)

        weights = torch.tensor([fwd_decay_ratio**i for i in range(len(act_cache))])
        weights = weights / weights.sum()

        weighted_act = (act_cache * weights.unsqueeze(1)).sum(dim=0)

        return weighted_act

    @staticmethod
    def convert_old_state_dict(state_dict):
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_k = k.replace("module.", "")
            else:
                new_k = k

            if not new_k.startswith("model."):
                new_k = "model." + new_k

            new_state_dict[new_k] = state_dict[k]
        return new_state_dict

    def _get_default_calvin_config(self):
        return {
            "type": "DiskCalvinDataset",
            "data_dir": "CALVIN/task_ABCD_D/val",
            "c_act_scaler": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        }

    def add_element_to_queue(self, q: Queue, element):
        while q.qsize() >= q.maxsize:
            q.get()
        q.put(element)

    def get_history(self, q: Queue, pad: Literal["zero", "first"] = "zero"):
        queue_list = list(q.queue)
        if len(queue_list) == 0:
            return queue_list, None
        history_type = self.configs["act_head"].get("history_type", "pre")
        if history_type == "pre":
            pad_len = 0
        else:
            raise ValueError(f"Unsupported history type {history_type}")
        element = queue_list[0]
        if pad == "zero":
            if isinstance(element, torch.Tensor):
                element = torch.zeros_like(element)
            elif isinstance(element, np.ndarray):
                element = np.zeros_like(element)
            else:
                raise ValueError("This type is not supported")
            queue_list = [element for _ in range(pad_len)] + queue_list
        else:
            if isinstance(element, torch.Tensor):
                pad_list = [element.clone() for _ in range(pad_len)]
            elif isinstance(element, np.ndarray):
                pad_list = [deepcopy(element) for _ in range(pad_len)]
            queue_list = pad_list + queue_list
        pad_mask = np.ones(q.maxsize, dtype=bool)
        pad_mask[:pad_len] = False
        return queue_list, pad_mask

    def preprocess(self, obs, lang, mode="continuous"):
        obs["image"]["camera_0"] = (
            obs["image"]["camera_0"].squeeze().cpu().detach()
        )  # (224, 224, 3)
        obs["image"]["camera_1"] = obs["image"]["camera_1"].squeeze().cpu().detach()
        # preprocess static cam image
        image = obs["image"]["camera_1"].numpy()
        image = Image.fromarray(image)
        image_x = self.image_preprocess([image]).unsqueeze(0)  # (1, 1, 3, 224, 224)

        gripper_x = None
        if "camera_0" in obs["image"]:
            gripper = obs["image"]["camera_0"].numpy()
            gripper = Image.fromarray(gripper)
            gripper_x = self.image_preprocess([gripper]).unsqueeze(0)
            gripper_x = gripper_x.to(self.device).to(self.dtype)

        if self.configs["act_head"].get("history_type", "post") == "pre":
            self.add_element_to_queue(self.vision_queue, image_x)
            image_x, _ = self.get_history(self.vision_queue, pad="first")
            image_x = torch.concatenate(image_x, dim=1)

            if gripper_x is not None:
                self.add_element_to_queue(self.vision_gripper_queue, gripper_x)
                gripper_x, _ = self.get_history(self.vision_gripper_queue, pad="first")
                gripper_x = (
                    torch.concatenate(gripper_x, dim=1).to(self.device).to(self.dtype)
                )

        if mode == "discrete":
            if "llava" in self.policy.configs:
                model_name = self.policy.configs["llava"]
            elif "qwen" in self.policy.configs:
                model_name = "qwen"
            else:
                # model_name = self.policy.configs['llm']['pretrained_model_name_or_path']
                model_name = self.policy.configs["model"]

            prompt_builder = get_prompt_builder(
                model_name, bos=self.tokenizer.bos_token, eos=self.tokenizer.eos_token
            )

            conversation = [
                {
                    "from": "human",
                    "value": (
                        f"What action should the robot take to {lang}?"
                        if self.act_step == 1
                        else f"What {self.act_step} step actions should the robot take to {lang}?"
                    ),
                },
                {"from": "gpt", "value": ""},
            ]

            input_ids = []
            for turn in conversation:
                prompt_builder.add_turn(turn["from"], turn["value"])

            input_ids = torch.tensor(
                list(
                    self.tokenizer(
                        prompt_builder.get_prompt(), add_special_tokens=True
                    ).input_ids
                )
            )
            if self.tokenizer.eos_token is not None:
                input_ids = input_ids[:-1]

            text_x = input_ids.unsqueeze(0)
            mask = torch.full((1, text_x.shape[-1]), True, dtype=torch.bool)
        else:
            text_x, mask = self.text_preprocess([lang])

        return (
            image_x.to(self.device).to(self.dtype),
            gripper_x,
            text_x.to(self.device),
            mask.to(self.device),
        )

    def reset(self):
        print("reset model now!!!!!!!!")
        if hasattr(self.model.model, "lm_head"):
            self.model.model.lm_head.hidden_state = None
            self.model.model.lm_head.history_memory = []
        if hasattr(self.model.model, "act_head"):
            self.model.model.act_head.hidden_state = None
            self.model.model.act_head.history_memory = []

        self.rgb_list = []
        self.hand_rgb_list = []
        self.rollout_step_counter = 0
        self.action_hist_list = []

        while not self.vision_queue.empty():
            self.vision_queue.get()
        while not self.vision_gripper_queue.empty():
            self.vision_gripper_queue.get()
        while not self.action_queue.empty():
            self.action_queue.get()

    def __call__(self, obs, **kwargs):
        """
        Forward pass through the model.
        :param obs: Observation input to the model.
        :return: Model output.
        """
        """Step function."""
        input_dict = dict()
        image_x, gripper_x, text_x, mask = self.preprocess(
            obs, self.task_name, self.action_space
        )

        input_dict["rgb"] = image_x
        input_dict["hand_rgb"] = gripper_x
        input_dict["text"] = text_x
        input_dict["text_mask"] = mask

        if self.action_space == "discrete":
            input_dict["instr_and_action_ids"] = text_x
            input_dict["instr_and_action_mask"] = mask

        start = time.time()
        with torch.no_grad():
            action = self.policy.inference_step(input_dict)["action"]
        # print("original action from model: ", action)
        print("**** RoboVLMs inference time: ", time.time() - start)

        if self.action_space != "discrete":
            if action[0].ndim == action[1].ndim + 1:
                action = (action[0], action[1].unsqueeze(2))
            action = torch.cat(
                [action[0], (torch.nn.functional.sigmoid(action[1]) > 0.5).float()],
                dim=-1,
            )

        if isinstance(action, tuple):
            action = torch.cat([action[0], action[1]], dim=-1)

        if isinstance(action, np.ndarray):
            action = torch.from_numpy(action)

        if action.ndim == 2:
            action = action.unsqueeze(1)

        if action.ndim == 3:
            action = action.unsqueeze(1)

        action = action.detach().cpu()

        if self.tcp_rel:
            robot_obs = (
                torch.from_numpy(obs["robot_obs"])
                .unsqueeze(0)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(1, self.window_size, self.fwd_pred_next_n, 1)
            )
            action = tcp_to_world_frame(action, robot_obs)

        if not self.use_act_chunk:
            print("ensemble action!!!")
            action = self.ensemble_action(action)

        if isinstance(action, torch.Tensor):
            action = (
                action.squeeze()
            )  # (fwd, 7) for action chunk, (7,) for ensembled action
            if not self.use_act_chunk:
                action = action.unsqueeze(0)

        if self.configs.get("use_mu_law", False):
            from robovlms.data.data_utils import inverse_mu_law_companding

            action = inverse_mu_law_companding(
                action, self.configs.get("mu_val", 255), maintain_last=True
            )

        # unnorm action
        if self.configs.get("norm_action", False):
            from maniunicon.utils.vla_utils import unnoramalize_action_perdim

            if isinstance(action, tuple):
                action = (
                    unnoramalize_action_perdim(
                        action[0],
                        np.array(self.configs["norm_min"]),
                        np.array(self.configs["norm_max"]),
                    ),
                    action[1],
                )
            else:
                action = unnoramalize_action_perdim(
                    action,
                    np.array(self.configs["norm_min"]),
                    np.array(self.configs["norm_max"]),
                    maintain_last=True,
                )

        self.rollout_step_counter += 1
        if isinstance(action, torch.Tensor):
            action = action.numpy()

        # convert rot from euler to quat
        from maniunicon.utils.vla_utils import euler_pose_to_quat

        N, A = action.shape
        action = np.concatenate(
            [
                euler_pose_to_quat(action[..., :-1].reshape(-1, A - 1)).reshape(N, -1),
                action[..., -1:],
            ],
            axis=-1,
        )
        print(f"step {self.rollout_step_counter} action {action}")

        return action
