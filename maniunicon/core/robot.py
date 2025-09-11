import torch.multiprocessing as mp
import threading
import time
import traceback
import numpy as np
from multiprocessing.synchronize import Event
from loop_rate_limiters import RateLimiter
from scipy.spatial.transform import Rotation

from maniunicon.robot_interface.base import RobotInterface
from maniunicon.utils.shared_memory.shared_storage import SharedStorage, RobotAction
from maniunicon.utils.pose_trajectory_interpolator import PoseTrajectoryInterpolator
from maniunicon.utils.timestamp_accumulator import TimestampAlignedBuffer


class Robot(mp.Process):
    """Process that handles both robot state receiving and action sending using separate threads."""

    def __init__(
        self,
        robot_interface: RobotInterface,
        shared_storage: SharedStorage,
        control_frequency: float = 100.0,  # Hz
        state_update_frequency: float = 50.0,  # Hz
        update_state: bool = True,
        use_interpolator: bool = True,
        max_pos_speed: float = 0.25,
        max_rot_speed: float = 0.5,  # need to be tuned
        reset_event: Event = None,
        ee_trans_mat: np.ndarray | list = np.eye(3),  # used by VR controller
        name: str = "Robot",
        synchronized: bool = False,
        warn_on_late: bool = True,
        workspace_bounds: dict = None,
    ):
        super().__init__(name=name)
        self.robot_interface = robot_interface
        self.shared_storage = shared_storage
        self.control_frequency = control_frequency
        self.state_update_frequency = state_update_frequency
        self.use_interpolator = use_interpolator
        self.last_action_timestamp = 0.0
        self.max_pos_speed = max_pos_speed
        self.max_rot_speed = max_rot_speed
        self.ee_trans_mat = (
            np.array(ee_trans_mat) if isinstance(ee_trans_mat, list) else ee_trans_mat
        )  # used by VR controller
        self.reset_event = reset_event
        # do not update state storage in robot side if we are in replay mode
        self.update_state = update_state
        self.state_thread = None
        self.state_record_buffer = None
        self.action_record_buffer = None
        self.synchronized = synchronized
        self.warn_on_late = warn_on_late
        self.workspace_bounds = workspace_bounds

    def connect(self):
        return self.robot_interface.connect()

    def disconnect(self):
        return self.robot_interface.disconnect()

    def reset_to_init(self):
        """Reset the robot to the home position."""
        self.robot_interface.reset_to_init()

    def move_to_joint_positions(self, joint_positions: np.ndarray):
        """Move the robot to the given joint positions."""
        self.robot_interface.move_to_joint_positions(joint_positions)

    def _clip_action_to_bounds(self, action: RobotAction) -> RobotAction:
        """Clip action to stay within workspace bounds."""
        if self.workspace_bounds is None or not self.workspace_bounds.get(
            "enabled", False
        ):
            return action

        # Create a copy of the action to avoid modifying the original
        clipped_action = RobotAction(
            control_mode=action.control_mode,
            timestamp=action.timestamp,
            target_timestamp=action.target_timestamp,
            tcp_position=action.tcp_position.copy(),
            tcp_orientation=action.tcp_orientation.copy(),
            joint_positions=(
                action.joint_positions.copy()
                if action.joint_positions is not None
                else None
            ),
            gripper_state=action.gripper_state.copy(),
        )

        # Get TCP position and orientation for bounds checking
        tcp_position = clipped_action.tcp_position
        tcp_orientation = clipped_action.tcp_orientation

        # For joint control mode, use forward kinematics if TCP pose is not available
        if action.control_mode == "joint" and (
            np.isnan(tcp_position).any() or np.isnan(tcp_orientation).any()
        ):
            tcp_position, tcp_orientation = self.robot_interface.forward_kinematics(
                clipped_action.joint_positions
            )

        # Clip position bounds
        pos_bounds = self.workspace_bounds.get("position", {})
        if pos_bounds.get("x_min") is not None:
            tcp_position[0] = max(tcp_position[0], pos_bounds["x_min"])
        if pos_bounds.get("x_max") is not None:
            tcp_position[0] = min(tcp_position[0], pos_bounds["x_max"])
        if pos_bounds.get("y_min") is not None:
            tcp_position[1] = max(tcp_position[1], pos_bounds["y_min"])
        if pos_bounds.get("y_max") is not None:
            tcp_position[1] = min(tcp_position[1], pos_bounds["y_max"])
        if pos_bounds.get("z_min") is not None:
            tcp_position[2] = max(tcp_position[2], pos_bounds["z_min"])
        if pos_bounds.get("z_max") is not None:
            tcp_position[2] = min(tcp_position[2], pos_bounds["z_max"])

        # Clip orientation bounds
        ori_bounds = self.workspace_bounds.get("orientation", {})
        if any(
            ori_bounds.get(key) is not None
            for key in ["x_min", "x_max", "y_min", "y_max", "z_min", "z_max"]
        ):
            # Convert quaternion to euler angles for bounds checking
            rotation = Rotation.from_quat(tcp_orientation)
            roll, pitch, yaw = rotation.as_euler("xyz", degrees=False)

            # Clip roll (X rotation)
            if ori_bounds.get("x_min") is not None:
                roll = max(roll, ori_bounds["x_min"])
            if ori_bounds.get("x_max") is not None:
                roll = min(roll, ori_bounds["x_max"])

            # Clip pitch (Y rotation)
            if ori_bounds.get("y_min") is not None:
                pitch = max(pitch, ori_bounds["y_min"])
            if ori_bounds.get("y_max") is not None:
                pitch = min(pitch, ori_bounds["y_max"])

            # Clip yaw (Z rotation)
            if ori_bounds.get("z_min") is not None:
                yaw = max(yaw, ori_bounds["z_min"])
            if ori_bounds.get("z_max") is not None:
                yaw = min(yaw, ori_bounds["z_max"])

            # Convert back to quaternion
            clipped_rotation = Rotation.from_euler(
                "xyz", [roll, pitch, yaw], degrees=False
            )
            tcp_orientation = clipped_rotation.as_quat()

        # Update the action with clipped values
        clipped_action.tcp_position = tcp_position
        clipped_action.tcp_orientation = tcp_orientation

        # For joint control mode, always compute IK back to joint space after clipping
        if action.control_mode == "joint":
            # Use inverse kinematics to get joint positions for clipped pose
            try:
                clipped_joint_positions = self.robot_interface.inverse_kinematics(
                    tcp_position, tcp_orientation, clipped_action.joint_positions
                )
                if clipped_joint_positions is not None:
                    clipped_action.joint_positions = clipped_joint_positions
            except Exception as e:
                print(f"Warning: Could not compute IK for clipped pose: {e}")
                # Keep original joint positions if IK fails

        return clipped_action

    def _state_receiver_thread(self):
        """Thread that receives and updates robot state."""
        rate = RateLimiter(
            frequency=self.state_update_frequency,
            warn=self.warn_on_late,
            name="state_receiver",
        )
        while self.shared_storage.is_running.value:
            try:
                # Get state from robot
                state = self.robot_interface.get_state()

                if self.shared_storage.is_recording.value:
                    if self.state_record_buffer is None:
                        # XXX(zbzhu): For a multi-step robot policy, the action buffer should set overwrite to True.
                        # However, the current overwrite buffer has a bug where some entries might remain unfilled and thus
                        # contain incorrect values. Since we currently only record single-step teleop controller data, the action
                        # buffer is set to not allow overwrite.
                        self.state_record_buffer = TimestampAlignedBuffer(
                            self.shared_storage.record_start_time.value,
                            self.shared_storage.record_dt.value,
                            self.shared_storage.max_record_steps,
                            overwrite=False,
                        )
                    self.state_record_buffer.add(
                        state.model_dump(), timestamp=state.timestamp.item()
                    )
                else:
                    if self.state_record_buffer is not None:
                        self._dump_state_data()
                        self.state_record_buffer = None

                # Update shared memory
                self.shared_storage.write_state(state)

                # Check for errors
                if self.robot_interface.is_error():
                    self.shared_storage.error_state.value = True
                    break

            except Exception as e:
                print(f"Error in state receiver thread: {e}")
                traceback.print_exc()
                self.shared_storage.error_state.value = True
                break

            rate.sleep()

    def _dump_state_data(self):
        """Dump state data to file."""
        record_dir = self.shared_storage.get_record_dir()
        if not record_dir:
            print("Record directory not set, skipping state data dump")
            return

        self.state_record_buffer.dump(name="state", dir=record_dir)

    def _dump_action_data(self):
        """Dump action data to file."""
        record_dir = self.shared_storage.get_record_dir()
        if not record_dir:
            print("Record directory not set, skipping action data dump")
            return

        self.action_record_buffer.dump(name="action", dir=record_dir)

    def _reset_interpolator(self):
        curr_t = time.monotonic()
        last_waypoint_time = curr_t
        curr_state = None
        while curr_state is None:
            curr_state = self.shared_storage.read_state()
            time.sleep(0.1)
        rot = Rotation.from_quat(curr_state.tcp_orientation)
        rotvec = rot.as_rotvec()
        curr_pose = np.concatenate([curr_state.tcp_position, rotvec], axis=-1)
        pose_interp = PoseTrajectoryInterpolator(
            times=[curr_t],
            poses=[curr_pose],
        )
        return pose_interp, last_waypoint_time

    def run(self):
        if not self.robot_interface.connect():
            self.shared_storage.error_state.value = True
            return

        self.robot_interface.reset_to_init()
        if self.update_state:
            # Start state receiver thread
            self.state_thread = threading.Thread(
                target=self._state_receiver_thread, name="state_receiver"
            )
            self.state_thread.start()

        if self.use_interpolator:
            pose_interp, last_waypoint_time = self._reset_interpolator()
        else:
            executing_actions = []

        last_action = None
        actions = []
        rate = RateLimiter(
            frequency=self.control_frequency,
            warn=self.warn_on_late,
            name="robot_control",
        )
        is_executing_actions = False
        while self.shared_storage.is_running.value:
            try:
                if self.reset_event is not None and self.reset_event.is_set():
                    print("Reset event detected, driving robot to init position")
                    self.reset_to_init()
                    last_action = None
                    actions = []

                    # Reset synchronization state during reset
                    if self.synchronized:
                        self.shared_storage.robot_ready.set()
                        self.shared_storage.policy_ready.clear()

                    # !!!IMPORTANT!!!: wait for the latest state to be updated
                    # also wait for the policy to sync state and clear the previous actions
                    time.sleep(0.5)
                    if self.use_interpolator:
                        pose_interp, last_waypoint_time = self._reset_interpolator()
                    else:
                        executing_actions = []
                    self.reset_event.clear()

                # Synchronization logic: wait for policy to be ready for execution
                if (
                    self.synchronized
                    and (self.reset_event is None or not self.reset_event.is_set())
                    and not is_executing_actions
                ):
                    self.shared_storage.policy_ready.wait()
                    # Clear policy_ready immediately after receiving the signal to complete handshake
                    self.shared_storage.policy_ready.clear()

                t_now = time.monotonic()
                if self.use_interpolator and (
                    last_action is None or last_action.control_mode == "cartesian"
                ):
                    # get interpolated action at current time
                    if t_now > pose_interp.times[-1]:
                        action = None
                    else:
                        pose_command = pose_interp(t_now)
                        # Convert rotation vector back to quaternion [x,y,z,w]
                        rot = Rotation.from_rotvec(pose_command[3:])
                        quat = rot.as_quat()
                        action = RobotAction(
                            control_mode="cartesian",
                            timestamp=time.time(),
                            tcp_position=pose_command[:3],
                            tcp_orientation=quat,
                            gripper_state=(
                                last_action.gripper_state
                                if last_action
                                else np.array([0.0])
                            ),
                        )
                else:
                    if len(executing_actions) > 0:
                        while (
                            executing_actions[0].target_timestamp < t_now
                            and len(executing_actions) > 1
                        ):
                            executing_actions = executing_actions[1:]
                        action = executing_actions[0]
                    else:
                        action = None

                # Only send if it's a new action
                if (
                    action is not None
                    and action.timestamp >= self.last_action_timestamp
                ):
                    # Apply workspace bounds clipping
                    clipped_action = self._clip_action_to_bounds(action)

                    # Validate action before sending
                    if self.robot_interface.validate_action(clipped_action):
                        if not self.robot_interface.send_action(clipped_action):
                            self.shared_storage.error_state.value = True
                            break
                        self.last_action_timestamp = action.timestamp
                        if not self.use_interpolator:
                            last_action = clipped_action
                    else:
                        print("Invalid action detected, stopping robot")
                        self.robot_interface.stop()
                        self.shared_storage.error_state.value = True
                        break

                # Get latest action
                actions = self.shared_storage.read_all_action()
                if self.shared_storage.is_recording.value:
                    if self.action_record_buffer is None:
                        self.action_record_buffer = TimestampAlignedBuffer(
                            self.shared_storage.record_start_time.value,
                            self.shared_storage.record_dt.value,
                            self.shared_storage.max_record_steps,
                            overwrite=False,
                        )
                else:
                    if self.action_record_buffer is not None:
                        self._dump_action_data()
                        self.action_record_buffer = None

                if len(actions) > 0:
                    # print(f"Received {len(actions)} actions")
                    # Clear robot_ready only when there are actions to process
                    if self.synchronized and (
                        self.reset_event is None or not self.reset_event.is_set()
                    ):
                        self.shared_storage.robot_ready.clear()
                        is_executing_actions = True

                    if self.action_record_buffer is not None:
                        for action in actions:
                            self.action_record_buffer.add(
                                action.model_dump(), timestamp=action.target_timestamp
                            )

                    if self.synchronized:
                        current_time = time.time()
                        # Recompute all target timestamps relative to current time
                        # to make sure all actions get executed
                        for action in actions:
                            time_offset = action.target_timestamp - actions[0].timestamp
                            action.target_timestamp = current_time + time_offset

                    if (
                        self.use_interpolator
                        and actions[-1].control_mode == "cartesian"
                        and actions[-1].target_timestamp is not None
                    ):
                        last_action = actions[-1]
                        # update interpolator using all fetched actions
                        for action in actions:
                            if np.isnan(action.tcp_position).any():
                                (
                                    action.tcp_position,
                                    action.tcp_orientation,
                                ) = self.robot_interface.forward_kinematics(
                                    action.joint_positions
                                )
                            # Convert quaternion [x,y,z,w] to rotation vector
                            rot = Rotation.from_quat(action.tcp_orientation)
                            rotvec = rot.as_rotvec()
                            target_pose = np.concatenate(
                                [action.tcp_position, rotvec], axis=-1
                            )
                            target_time = action.target_timestamp
                            target_time = time.monotonic() - time.time() + target_time
                            curr_time = t_now + rate.period
                            pose_interp = pose_interp.schedule_waypoint(
                                pose=target_pose,
                                time=target_time,
                                max_pos_speed=self.max_pos_speed,
                                max_rot_speed=self.max_rot_speed,
                                curr_time=curr_time,
                                last_waypoint_time=last_waypoint_time,
                            )
                            last_waypoint_time = target_time
                    else:
                        executing_actions = actions

                # Synchronization logic: signal robot ready when actions are completed
                if (
                    self.synchronized
                    and (self.reset_event is None or not self.reset_event.is_set())
                    and is_executing_actions
                ):
                    # Check if all actions are completed
                    if (
                        self.use_interpolator
                        and (action is None or t_now > pose_interp.times[-1])
                    ) or (not self.use_interpolator and action is None):
                        # Actions are completed, signal robot ready for next inference
                        self.shared_storage.robot_ready.set()
                        is_executing_actions = False

                rate.sleep()

            except Exception as e:
                print(f"Error in robot action thread: {e}")
                print(traceback.format_exc())
                self.shared_storage.error_state.value = True
                # Reset synchronization state on error
                if self.synchronized:
                    self.shared_storage.robot_ready.set()
                    self.shared_storage.policy_ready.clear()
                break

        # Cleanup
        self.robot_interface.stop()
        self.robot_interface.disconnect()
        if self.state_thread is not None:
            self.state_thread.join()

    def stop(self):
        """Stop the controller process and all threads."""
        self.shared_storage.is_running.value = False
        self.join()
