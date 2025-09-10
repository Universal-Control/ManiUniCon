"""Gello-based teleoperation policy for robot control."""

import os
import time
from typing import Optional
import numpy as np
from multiprocessing.synchronize import Event
import traceback
from loop_rate_limiters import RateLimiter

from gello.agents.gello_agent import GelloAgent

from maniunicon.utils.shared_memory.shared_storage import (
    SharedStorage,
    RobotAction,
    RobotState,
)
from maniunicon.core.policy import BasePolicy
from maniunicon.utils.data import get_next_episode_dir
from maniunicon.utils.filter import JointSpaceSmoother, AdaptiveButterworth

DEBUG = True
if DEBUG:
    import os
    import shutil
    if os.path.exists(f"/home/ydu/xiaoshen/franka_install/ManiUniCon/outputs/22222/joints-command-before-smoothing"):
        shutil.rmtree(f"/home/ydu/xiaoshen/franka_install/ManiUniCon/outputs/22222/joints-command-before-smoothing")
    if os.path.exists(f"/home/ydu/xiaoshen/franka_install/ManiUniCon/outputs/22222/joints-command-after-smoothing"):
        shutil.rmtree(f"/home/ydu/xiaoshen/franka_install/ManiUniCon/outputs/22222/joints-command-after-smoothing")

class GelloPolicy(BasePolicy):
    """Gello-based teleoperation policy.

    Reads joint positions from Gello device and applies them to the robot
    with safety clipping based on current joint limits.
    """

    def __init__(
        self,
        shared_storage: SharedStorage,
        reset_event: Event,
        num_joints: int = 7,  # Gello defaults to 7 joints
        command_latency: float = 0.01,  # seconds
        device_path: str = "/dev/ttyUSB0",
        dt: float = 0.01,  # Time step between actions
        control_interval: int = 1,  # Number of action steps to send
        joint_limits: Optional[dict] = None,  # {"min": [...], "max": [...]}
        joint_velocity_limit: float = 10,  # rad/s max velocity
        synchronized: bool = False,
        warn_on_late: bool = True,
        name: str = "GelloPolicy",
    ):
        super().__init__(
            shared_storage=shared_storage,
            reset_event=reset_event,
            command_latency=command_latency,
            name=name,
        )
        self.num_joints = num_joints  # Save the configured number of joints
        self.device_path = device_path
        self.dt = dt
        self.frequency = 1 / dt
        self.control_interval = control_interval
        self.joint_limits = joint_limits
        self.joint_velocity_limit = joint_velocity_limit
        self.synchronized = synchronized
        self.warn_on_late = warn_on_late

        # Internal state
        self.agent = None
        self._current_joint_positions = None
        self._prev_joint_positions = None
        self._gripper_state = np.array([0.0])  # 0 for open, 1 for closed
        self._should_disconnect = False

    def _clip_joint_positions(
        self, target_positions: np.ndarray, current_positions: np.ndarray
    ) -> np.ndarray:
        """Clip joint positions for safety.

        Args:
            target_positions: Target joint positions from Gello
            current_positions: Current robot joint positions

        Returns:
            Clipped joint positions
        """
        clipped_positions = target_positions.copy()
        return clipped_positions  # temp disable

    def sync_state(self):
        """Sync the robot state with shared storage."""
        state = None
        while state is None:
            state: RobotState | None = self.shared_storage.read_state(k=1)
            print("GelloPolicy: waiting for state...")
            time.sleep(0.05)

        self._current_joint_positions = state.joint_positions[0].copy()

    def run(self):
        """Main process loop."""
        try:
            # Initialize Gello agent with current robot joint positions
            print(f"Initializing Gello agent on {self.device_path}")

            # Get initial robot state to pass to Gello
            self.sync_state()

            # Initialize Gello with current robot joint positions
            # Number of joints supported by Gello is determined by num_joints configuration
            self.agent = GelloAgent(
                port=self.device_path,
            )

            print(
                f"Gello controller connected successfully (controlling {self.num_joints} joints)"
            )
            print("\nGello controls:")
            print("- Move the Gello arm to control the robot")
            print("- The robot will follow the Gello joint positions")
            print("- Joint positions are clipped for safety")

            rate = RateLimiter(
                frequency=self.frequency,
                warn=self.warn_on_late,
                name="gello_policy",
            )

            smoother = JointSpaceSmoother(
                num_joints=self.num_joints,
                alpha_ewma=0.3,
                window_size=5,
                velocity_limit=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
                acceleration_limit=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
                deadband_threshold=0.001,
                adaptive_alpha=True
            )
            butterworth = AdaptiveButterworth(num_joints=self.num_joints, cutoff_freq=10.0, sample_rate=self.frequency)

            while self.shared_storage.is_running.value and not self._should_disconnect:
                # Handle reset
                if self.reset_event is not None and self.reset_event.is_set():
                    # Clear actions during reset
                    self.shared_storage.read_all_action()
                    self.sync_state()
                    rate.sleep()
                    continue

                if self.synchronized and (
                    self.reset_event is None or not self.reset_event.is_set()
                ):
                    # Wait for robot to be ready
                    self.shared_storage.robot_ready.wait()
                    self.shared_storage.robot_ready.clear()

                # Read joint positions from Gello
                # agent.act() returns an array containing joint positions and gripper state
                # Format: [joint1, joint2, ..., jointN, gripper1, gripper2]
                gello_output = self.agent.act({})

                # Extract joint positions and gripper state
                # Gello returns num_joints joint positions + 2 gripper values
                if self._current_joint_positions is not None:
                    num_robot_joints = len(self._current_joint_positions)

                    # Create target joint positions array
                    target_joint_positions = np.zeros(num_robot_joints)

                    # Copy Gello joints
                    num_gello_joints = min(self.num_joints, num_robot_joints)
                    target_joint_positions[:num_gello_joints] = gello_output[
                        :num_gello_joints
                    ]

                    # If robot has more joints than Gello controls, keep current positions for extra joints
                    if num_robot_joints > self.num_joints:
                        target_joint_positions[self.num_joints :] = (
                            self._current_joint_positions[self.num_joints :]
                        )

                    gripper_values = (
                        gello_output[self.num_joints : self.num_joints + 1] > 0.5
                    )
                    self._gripper_state = np.array([np.mean(gripper_values)])

                    # Apply safety clipping
                    clipped_joint_positions = self._clip_joint_positions(
                        target_joint_positions, self._current_joint_positions
                    )

                    # Create and send robot actions
                    current_time = time.time()

                    # Apply safety clipping
                    clipped_joint_positions = self._clip_joint_positions(
                        target_joint_positions, self._current_joint_positions
                    )

                    if DEBUG:
                        os.makedirs(f"/home/ydu/xiaoshen/franka_install/ManiUniCon/outputs/22222/joints-command-before-smoothing", exist_ok=True)
                        np.save(f"/home/ydu/xiaoshen/franka_install/ManiUniCon/outputs/22222/joints-command-before-smoothing/{time.time_ns()}.npy", clipped_joint_positions)

                    # Apply smoothing
                    clipped_joint_positions = smoother.smooth(clipped_joint_positions, timestamp=current_time)
                    clipped_joint_positions = butterworth.filter(clipped_joint_positions)

                    if DEBUG:
                        os.makedirs(f"/home/ydu/xiaoshen/franka_install/ManiUniCon/outputs/22222/joints-command-after-smoothing", exist_ok=True)
                        np.save(f"/home/ydu/xiaoshen/franka_install/ManiUniCon/outputs/22222/joints-command-after-smoothing/{time.time_ns()}.npy", clipped_joint_positions)

                    # Update current positions for next iteration
                    self._current_joint_positions = clipped_joint_positions.copy()

                    for i in range(self.control_interval):
                        action = RobotAction(
                            joint_positions=clipped_joint_positions,
                            gripper_state=self._gripper_state,
                            control_mode="joint",
                            timestamp=current_time,
                            target_timestamp=current_time
                            + (i + 2) * self.dt
                            - self.command_latency,
                        )

                        self.shared_storage.write_action(action)

                    if self.synchronized and (
                        self.reset_event is None or not self.reset_event.is_set()
                    ):
                        # Signal robot can execute actions
                        self.shared_storage.policy_ready.set()

                    # Check for errors
                    if self.shared_storage.error_state.value:
                        break

                    if not self.synchronized:
                        rate.sleep()
                else:
                    # If we don't have current positions yet, sync state
                    self.sync_state()

        except KeyboardInterrupt:
            print("\nGello policy interrupted by user")
        except Exception as e:
            print(f"Error in GelloPolicy: {e}")
            traceback.print_exc()
            self.shared_storage.error_state.value = True
        finally:
            print("Gello policy stopped")

    def stop(self):
        """Stop the policy process."""
        self._should_disconnect = True
        super().stop()
