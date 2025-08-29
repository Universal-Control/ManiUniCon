import os
import time
import numpy as np
from typing import Callable, Any, Optional
import hydra
import torch
from multiprocessing.synchronize import Event
import traceback
from pynput import keyboard

from maniunicon.utils.data import get_next_episode_dir
from maniunicon.utils.shared_memory.shared_storage import (
    SharedStorage,
    MultiCameraData,
)
from maniunicon.core.policy import BasePolicy


def precise_wait(t_end: float, slack_time: float = 0.001, time_func=time.monotonic):
    t_start = time_func()
    t_wait = t_end - t_start
    if t_wait > 0:
        t_sleep = t_wait - slack_time
        if t_sleep > 0:
            time.sleep(t_sleep)
        while time_func() < t_end:
            pass
    return


def compile():
    print("global compile setting..")
    torch._dynamo.reset()
    torch._dynamo.config.cache_size_limit = 256
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.capture_dynamic_output_shape_ops = True


class TorchModelPolicy(BasePolicy):
    """Process of robot policy model."""

    def __init__(
        self,
        shared_storage: SharedStorage,
        reset_event: Event,
        model: Callable,
        obs_wrapper: Callable,
        act_wrapper: Callable,
        dt: float = 0.02,  # policy future actions dt
        name: str = "TorchModelPolicy",
        steps_per_inference: int = 1,
        infer_latency: float = 1.0 / 20,  # seconds
        frame_latency: float = 1.0 / 30,  # seconds
        use_real_time_chunking: bool = False,
        record_dir: Optional[str] = None,
        enable_recording: bool = False,
        device: str = "cpu",
        synchronized: bool = False,
        **kwargs: Any,
    ):
        super().__init__(
            shared_storage=shared_storage,
            reset_event=reset_event,
            name=name,
        )
        self.model = model
        self.obs_wrapper = obs_wrapper
        self.act_wrapper = act_wrapper
        self.dt = dt
        self.obs_horizon = model.network.observation_horizon
        self.steps_per_inference = steps_per_inference
        self.frame_latency = frame_latency
        self.use_real_time_chunking = use_real_time_chunking
        self.enable_recording = enable_recording
        self.infer_latency = infer_latency
        self.device = device
        self.synchronized = synchronized
        self.activate = False
        if self.use_real_time_chunking:
            self.prev_raw_actions = None
            self.prev_raw_timestamps = None

        self.record_dir = record_dir
        self._recording_active = False
        self._recording_key_pressed = False
        self._current_episode_dir = None
        self._listener = None

    def _on_press(self, key):
        """Handle key press events."""
        try:
            # Handle recording toggle on key press (not hold)
            if (
                key == keyboard.KeyCode.from_char("r")
                and not self._recording_key_pressed
            ):
                self._recording_key_pressed = True
                self._recording_active = not self._recording_active
                print(f"Recording: {'ON' if self._recording_active else 'OFF'}")
            # Handle dropping current episode
            elif key == keyboard.KeyCode.from_char("d") and self._recording_active:
                self._recording_active = False
                # First stop recording in shared storage
                self.shared_storage.clear_record_dir()
                # Then remove the directory if it exists
                if self._current_episode_dir and os.path.exists(
                    self._current_episode_dir
                ):
                    import shutil

                    shutil.rmtree(self._current_episode_dir)
                    print(
                        f"Dropped episode - Removed directory: {self._current_episode_dir}"
                    )
                self._current_episode_dir = None
                self.shared_storage.stop_record()
                print("Recording: OFF")
        except AttributeError:
            pass

    def _on_release(self, key):
        """Handle key release events."""
        try:
            # Reset recording key press state
            if key == keyboard.KeyCode.from_char("r"):
                self._recording_key_pressed = False
        except AttributeError:
            pass

    def run(self):
        """Main process loop."""
        try:
            compile()  # speedup model inference
            self.model = hydra.utils.instantiate(
                self.model, device=self.device, _recursive_=False
            )
            self.obs_wrapper = hydra.utils.instantiate(
                self.obs_wrapper,
                shared_storage=self.shared_storage,
                device=self.device,
            )
            self.act_wrapper = hydra.utils.instantiate(
                self.act_wrapper,
            )

            if self.enable_recording:
                self._listener = keyboard.Listener(
                    on_press=self._on_press, on_release=self._on_release
                )
                self._listener.start()

                while not self._recording_active:
                    print("Waiting for pressing 'r' to start recording...")
                    time.sleep(0.1)

            def _on_press(key):
                if key == keyboard.KeyCode.from_char("g"):
                    print("Start policy inference!")
                    self.activate = True

            start_listener = keyboard.Listener(on_press=_on_press)
            start_listener.start()

            start_delay = 1.0
            eval_t_start = time.time() + start_delay
            t_start = time.monotonic() + start_delay
            frame_latency = self.frame_latency
            precise_wait(eval_t_start - frame_latency, time_func=time.time)
            iter_idx = 0
            while self.shared_storage.is_running.value:
                # calculate timing
                if self.synchronized:
                    t_cycle_end = time.monotonic() + self.dt
                    iter_idx = int((t_cycle_end - t_start) / self.dt)
                else:
                    t_cycle_end = (
                        t_start + (iter_idx + self.steps_per_inference) * self.dt
                    )
                    iter_idx += self.steps_per_inference

                if self.reset_event is not None and self.reset_event.is_set():
                    self.shared_storage.read_all_action()
                    self.activate = False
                    precise_wait(t_cycle_end - frame_latency)
                    continue

                if not self.activate:
                    print("pressing 'g' to start policy inference...")
                    precise_wait(t_cycle_end - frame_latency)
                    continue

                # Synchronization logic: wait for robot to be ready before inference
                if self.synchronized and (
                    self.reset_event is None or not self.reset_event.is_set()
                ):
                    # wait for robot to be ready
                    self.shared_storage.robot_ready.wait()
                    self.shared_storage.robot_ready.clear()

                if self.enable_recording:
                    # Handle recording state changes
                    if (
                        self._recording_active
                        and not self.shared_storage.is_recording.value
                    ):
                        # Start recording
                        if self.record_dir is not None:
                            self._current_episode_dir = get_next_episode_dir(
                                self.record_dir
                            )
                            self.shared_storage.set_record_dir(
                                self._current_episode_dir
                            )
                            self.shared_storage.start_record(
                                start_time=time.time(),
                                dt=self.dt,
                            )
                            print(
                                f"Recording started - Episode: {os.path.basename(self._current_episode_dir)}"
                            )
                        else:
                            print(
                                "Recording directory not specified. Cannot start recording."
                            )
                            self._recording_active = False
                    elif (
                        not self._recording_active
                        and self.shared_storage.is_recording.value
                    ):
                        # Stop recording
                        self.shared_storage.stop_record()
                        if self._current_episode_dir:
                            print(
                                f"Recording stopped - Episode saved to: {self._current_episode_dir}"
                            )
                        else:
                            print("Recording stopped")
                        self._current_episode_dir = None
                        self._recording_active = False

                state = self.shared_storage.read_state(k=self.obs_horizon)
                camera: MultiCameraData = self.shared_storage.read_multi_camera(
                    k=self.obs_horizon
                )
                obs_time = state.timestamp[-1].item()

                if state is None:
                    print("Robot is not ready, skipping policy")
                    precise_wait(t_cycle_end - frame_latency)
                    continue
                if camera is None:
                    print("Camera is not ready, skipping policy")
                    precise_wait(t_cycle_end - frame_latency)
                    continue

                start = time.time()
                # Get latest obs
                obs: torch.Tensor = self.obs_wrapper(state=state, camera=camera)

                if self.use_real_time_chunking:
                    # if self.prev_raw_timestamps is not None:
                    #     print("time diff", obs_time - self.prev_raw_timestamps[0])
                    if self.prev_raw_actions is not None:
                        local_cond = np.zeros_like(self.prev_raw_actions)
                        overlap_idxes = np.where(self.prev_raw_timestamps > obs_time)[0]
                        num_overlap = overlap_idxes.shape[0]
                        hist_horizon = min(num_overlap, self.steps_per_inference)
                        local_cond[:, :, :hist_horizon] = self.prev_raw_actions[
                            :, :, overlap_idxes[:hist_horizon]
                        ]
                        local_cond = torch.from_numpy(local_cond).to(self.device)
                    else:
                        local_cond = None
                        hist_horizon = 0
                    actions, self.prev_raw_actions = self.act_wrapper(
                        self.model(
                            obs,
                            head_kwargs={
                                "local_cond": local_cond,
                                "hist_horizon": hist_horizon,
                            },
                        ),
                        timestamp=obs_time,
                        start_timestamp=eval_t_start,
                    )
                    self.prev_raw_timestamps = (
                        np.arange(self.prev_raw_actions.shape[2]) * self.dt + obs_time
                    )
                else:
                    actions = self.act_wrapper(
                        self.model(obs),
                        timestamp=obs_time,
                        start_timestamp=eval_t_start,
                        return_raw_actions=False,
                    )
                print(
                    "Obs wrapper + Policy Inference + Action wrapper: ",
                    time.time() - start,
                )

                # Check for errors
                if self.shared_storage.error_state.value:
                    break

                if self.enable_recording and not self._recording_active:
                    precise_wait(t_cycle_end - frame_latency)
                    continue

                for action in actions:
                    self.shared_storage.write_action(action)

                # Synchronization logic: signal policy ready for robot execution
                if self.synchronized and (
                    self.reset_event is None or not self.reset_event.is_set()
                ):
                    self.shared_storage.policy_ready.set()  # Signal robot can execute actions

                if not self.synchronized:
                    precise_wait(t_cycle_end - frame_latency)

        except Exception as e:
            print(f"Error in model: {e}")
            traceback.print_exc()
            self.shared_storage.error_state.value = True
            # Reset synchronization state on error
            if self.synchronized:
                self.shared_storage.robot_ready.set()
                self.shared_storage.policy_ready.clear()

    def stop(self):
        """Stop the policy process."""
        self.shared_storage.is_running.value = False
        self.join()
