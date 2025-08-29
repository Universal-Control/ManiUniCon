import signal
import sys
import time
from typing import Any, Dict
import torch.multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager

import hydra
from pynput import keyboard
from omegaconf import OmegaConf

from maniunicon.utils.shared_memory.shared_storage import SharedStorage


class RobotControlSystem:
    """Main class for managing the robot control system."""

    def __init__(
        self,
        data_cfg: Dict[str, Any],
        robot_cfg: Dict[str, Any],
        policy_cfg: Dict[str, Any],
        sensors_cfg: Dict[str, Any],
        max_record_steps: int = 500,
    ):
        # Create shared memory
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.start()
        self.shared_storage = SharedStorage(
            shm_manager=self.shm_manager,
            robot_state_config=data_cfg.robot_state_config,
            robot_action_config=data_cfg.robot_action_config,
            camera_config=data_cfg.get("camera_config", None),
            max_record_steps=max_record_steps,
        )
        self.reset_event = mp.Event()

        self.robot = hydra.utils.instantiate(
            robot_cfg,
            shared_storage=self.shared_storage,
            reset_event=self.reset_event,
        )

        self.policy = hydra.utils.instantiate(
            policy_cfg,
            shared_storage=self.shared_storage,
            reset_event=self.reset_event,
            _recursive_=False,
        )

        self.sensors = {
            name: hydra.utils.instantiate(
                sensor_cfg,
                shared_storage=self.shared_storage,
            )
            for name, sensor_cfg in sensors_cfg.items()
        }

        # Set up signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def start(self):
        """Start all processes."""
        print("Starting robot control system...")

        # Start processes in order
        print("Starting reset keyboard listener...")

        def _on_press(key):
            if key == keyboard.KeyCode.from_char("h"):
                print("Resetting robot to home position...")
                self.reset_event.set()

        keyboard_listener = keyboard.Listener(on_press=_on_press)
        keyboard_listener.start()
        print("Reset keyboard listener started")

        # Start the receiver (this will start all camera processes)
        for sensor_name, sensor in self.sensors.items():
            sensor.start()
            print(f"{sensor_name} started.")
        print("All sensors are started.")

        self.policy.start()
        print("Policy started.")

        self.robot.start()
        print("Robot started.")

        print("All processes started")

    def stop(self):
        """Stop all processes gracefully."""
        print("Stopping robot control system...")

        # Stop processes in reverse order
        self.robot.disconnect()
        self.robot.stop()
        self.policy.stop()
        for sensor in self.sensors.values():
            sensor.stop()

        self.shared_storage.is_running.value = False
        self.shm_manager.shutdown()

        print("All processes stopped")

    def _signal_handler(self, signum, frame):
        """Handle termination signals."""
        print(f"Received signal {signum}")
        self.stop()
        sys.exit(0)


@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="default",
)
def main(cfg):
    """Main entry point for the robot control system."""
    # Register custom OmegaConf resolver for mathematical expressions
    OmegaConf.register_new_resolver("eval", eval)

    # cfg = make_cfg(cfg_file)
    print(f"Configuration: {cfg}")
    mp.set_start_method("spawn")

    # Create and start control system
    control_system = RobotControlSystem(
        data_cfg=cfg.data,
        robot_cfg=cfg.robot,
        policy_cfg=cfg.policy,
        sensors_cfg=cfg.sensors,
        max_record_steps=cfg.max_record_steps,
    )

    try:
        control_system.start()

        # Keep main thread alive
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    except Exception as e:
        import traceback

        print(f"Error in main: {e}")
        traceback.print_exc()
        control_system.stop()
    finally:
        control_system.stop()


if __name__ == "__main__":
    main()
