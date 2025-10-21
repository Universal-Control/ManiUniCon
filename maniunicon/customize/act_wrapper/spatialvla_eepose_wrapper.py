import time
import torch
import numpy as np
from typing import Union

from maniunicon.utils.shared_memory.shared_storage import RobotAction


class SpatialVLAEEPoseWrapper:
    """
    A wrapper class for handling action for Robovlms.
    robot control mode: cartesian/ee pose
    """

    def __init__(
        self,
        action_horizon: int = 8,
        hist_action: int = 0,
        action_exec_latency: float = 0.01,
        control_mode: str = "cartesian",
        synchronized: bool = False,
        dt: float = 0.02,
        **kwargs,
    ):
        """
        Initialize the wrapper with history action and open loop steps.
        :param action_horizon: Number of open loop steps to execute.
        :param hist_action: Number of historical actions to consider.
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
        if actions.ndim == 4:
            # action: (bs, ws, fwd, action_dim) remove the batch dimension and history actions
            actions = actions[0, 0, self.hist_action : self.hist_action + self.action_horizon, :]
        elif actions.ndim == 2:
            # action: (fwd, action_dim)
            actions = actions[self.hist_action : self.hist_action + self.action_horizon, :]
        else:
            raise ValueError(f"Unsupported action shape: {actions.shape}. Expected 2D or 4D array.")

        print(np.sum(is_new), " new actions, ", len(actions), " total actions")
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
            # Convert to RobotAction
            robot_actions.append(
                RobotAction(
                    # FK is computed by the robot
                    tcp_position=action[:3],
                    tcp_orientation=action[3:7],
                    gripper_state=np.array([(action[7:8] > 0.5).astype(int)]),
                    control_mode=self.control_mode,
                    timestamp=timestamp,
                    target_timestamp=action_timestamps[idx],
                )
            )
        if return_raw_actions:
            return robot_actions, raw_actions
        else:
            return robot_actions
