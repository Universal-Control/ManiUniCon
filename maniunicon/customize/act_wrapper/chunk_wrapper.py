import time
import torch
import numpy as np
from typing import Union

from maniunicon.utils.shared_memory.shared_storage import RobotAction


class ActionChunkWrapper:
    """
    A unified wrapper class for handling action chunks for both joint and cartesian control modes.
    """

    def __init__(
        self,
        action_horizon: int = 8,
        hist_action: int = 0,
        action_exec_latency: float = 0.01,
        control_mode: str = "joint",
        synchronized: bool = False,
        dt: float = 0.02,
        **kwargs,
    ):
        """
        Initialize the wrapper with history action and open loop steps.
        :param action_horizon: Number of open loop steps to execute.
        :param hist_action: Number of historical actions to consider.
        :param control_mode: Control mode - either "joint" or "cartesian".
        """
        self.action_horizon = action_horizon
        self.hist_action = hist_action
        self.action_exec_latency = action_exec_latency
        self.control_mode = control_mode
        self.synchronized = synchronized
        self.dt = dt

    def __call__(
        self,
        actions: Union[torch.Tensor, np.ndarray],
        timestamp: float,
        start_timestamp: float,
        return_raw_actions: bool = True,
    ):
        robot_actions = []
        action_timestamps = np.arange(self.action_horizon) * self.dt + timestamp
        curr_time = time.time()
        # if synchronized execution is enabled, we do not need to check the action timestamp
        if self.synchronized:
            is_new = np.ones(action_timestamps.shape[0], dtype=bool)
        else:
            is_new = action_timestamps > (curr_time + self.action_exec_latency)

        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()

        raw_actions = actions.copy()
        actions = actions[
            0, 0, self.hist_action : self.hist_action + self.action_horizon, :
        ]  # remove the batch dimension and history actions

        if np.sum(is_new) == 0:
            # exceeded time budget, still do something
            actions = actions[-1:, :]
            next_step_idx = int(np.ceil((curr_time - start_timestamp) / self.dt))
            action_timestamp = start_timestamp + next_step_idx * self.dt
            print("No new actions !!! Check model inference time")
            action_timestamps = np.array([action_timestamp])
        else:
            actions = actions[is_new, :]
            action_timestamps = action_timestamps[is_new]

        for idx in range(len(actions)):
            action = actions[idx, :]

            # Handle different control modes
            if self.control_mode == "joint":
                # Joint control mode (from ppt_wrapper.py)
                robot_actions.append(
                    RobotAction(
                        joint_positions=action[:6],
                        gripper_state=np.array([(action[6:7] > 0.5).astype(int)]),
                        control_mode=self.control_mode,
                        timestamp=timestamp,
                        target_timestamp=action_timestamps[idx],
                    )
                )
            elif self.control_mode == "cartesian":
                # Cartesian control mode (from ppt_rgb_wrapper.py)
                robot_actions.append(
                    RobotAction(
                        tcp_position=action[:3],
                        tcp_orientation=action[3:7],
                        gripper_state=np.array([(action[7:8] > 0.5).astype(int)]),
                        control_mode=self.control_mode,
                        timestamp=timestamp,
                        target_timestamp=action_timestamps[idx],
                    )
                )
            else:
                raise ValueError(
                    f"Unsupported control mode: {self.control_mode}. Must be 'joint' or 'cartesian'."
                )

        if return_raw_actions:
            return robot_actions, raw_actions
        else:
            return robot_actions
