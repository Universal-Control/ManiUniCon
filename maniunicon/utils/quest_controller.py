import time
import numpy as np
from scipy.spatial.transform import Rotation as R

from oculus_reader.reader import OculusReader

import threading
from multiprocessing import Lock


def run_threaded_command(command, args=(), daemon=True):
    thread = threading.Thread(target=command, args=args, daemon=daemon)
    thread.start()
    return thread


def vec_to_reorder_mat(vec):
    X = np.zeros((len(vec), len(vec)))
    for i in range(X.shape[0]):
        ind = int(abs(vec[i])) - 1
        X[i, ind] = np.sign(vec[i])
    return X


def get_transformation(pos, rmat):
    transformation = np.eye(4)
    transformation[:3, :3] = rmat
    transformation[:3, 3] = pos
    return transformation


class VRPolicy:
    def __init__(
        self,
        ee_trans_mat: np.ndarray = np.eye(3),
        pos_scaling_factor: float = 1.0,
        rot_scaling_factor: float = 0.5,
        # Safety parameters
        max_delta_pos: float = 0.5,  # Maximum allowed position change per step (meters)
        max_delta_rot: float = 1.0,  # Maximum allowed rotation change per step (radians)
    ):
        # Transformation matrix from VR controller frame to robot's end-effector frame
        _tmp_transform_mat = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])

        # 60 degree
        self.R_ve = _tmp_transform_mat @ np.array(
            [
                [0, -1 / 2, np.sqrt(3) / 2],
                [-1, 0, 0],
                [0, -np.sqrt(3) / 2, -1 / 2],
            ]
        )
        # ee frame orientation may be different from robot world frame
        self.R_ve = ee_trans_mat @ self.R_ve

        # Velocity and action parameters
        self.pos_scaling_factor = pos_scaling_factor
        self.rot_scaling_factor = rot_scaling_factor

        # Safety parameters
        self.max_delta_pos = max_delta_pos
        self.max_delta_rot = max_delta_rot

        # Always use right controller for single arm
        self.controller_id = "r"

        # previous VR pose tracking
        self.previous_vr_rotation = None
        self.previous_vr_position = None

        # Accumulated robot pose
        self.accumulated_pose = None

        # Grip state tracking for automatic reset
        self.previous_grip_state = False

        # Gripper state tracking for toggle functionality
        self.gripper_state = 0  # 0 = open, 1 = closed
        self.previous_trigger_pressed = False

    def start(self):
        # Initialize Oculus reader
        self.oculus_reader = OculusReader()

        # Initialize state
        self._state_lock = Lock()
        self.reset_state()

        # Start the threaded state update
        run_threaded_command(self._update_internal_state)

    def reset_state(self):
        """Reset the internal state of the VR controller"""
        self._state = {
            "movement_enabled": False,
            "poses": {},
            "buttons": {},
        }

        # Reset tracking variables
        self.previous_vr_rotation = None
        self.previous_vr_position = None
        self.accumulated_pose = None

        # Reset grip state tracking
        self.previous_grip_state = False

        # Reset gripper state tracking
        self.gripper_state = 0
        self.previous_trigger_pressed = False

    def _update_internal_state(self, hz=50):
        """Continuously update the internal state from Oculus reader"""
        while True:
            time.sleep(1 / hz)
            transformations_and_buttons = (
                self.oculus_reader.get_transformations_and_buttons()
            )

            # Read Controller
            poses, buttons = transformations_and_buttons

            if poses == {}:
                with self._state_lock:
                    self._state["poses"] = {}
                continue

            # Save Info
            with self._state_lock:
                self._state["poses"] = poses
                self._state["buttons"] = buttons

    def _extract_vr_pose(self, vr_transform):
        """Extract position and rotation matrix from VR transformation matrix"""
        position = vr_transform[:3, 3]
        rotation = vr_transform[:3, :3]
        return position, rotation

    def _check_safety_limits(
        self, new_position, new_rotation, current_position, current_rotation
    ):
        """Check if the proposed movement violates safety limits"""
        # Calculate position and rotation deltas
        pos_delta = np.linalg.norm(new_position - current_position)
        # Use proper rotation distance calculation to avoid singularity issues
        rot_delta = (
            R.from_matrix(new_rotation) * R.from_matrix(current_rotation).inv()
        ).magnitude()

        # Check delta limits
        if pos_delta > self.max_delta_pos:
            print(
                f"SAFETY WARNING: Position delta {pos_delta:.4f}m exceeds limit {self.max_delta_pos}m"
            )
            return False

        if rot_delta > self.max_delta_rot:
            print(
                f"SAFETY WARNING: Rotation delta {rot_delta:.4f}rad exceeds limit {self.max_delta_rot}rad"
            )
            return False

        return True

    def _apply_safety_limits(
        self, new_position, new_rotation, current_position, current_rotation
    ):
        """Apply safety limits to the proposed movement"""
        # Calculate deltas
        pos_delta = new_position - current_position

        # Calculate rotation delta using proper rotation distance
        rot_delta_quat = (
            R.from_matrix(new_rotation) * R.from_matrix(current_rotation).inv()
        )
        rot_delta_rotvec = rot_delta_quat.as_rotvec()

        # Limit position delta
        pos_delta_norm = np.linalg.norm(pos_delta)
        if pos_delta_norm > self.max_delta_pos:
            pos_delta = pos_delta * (self.max_delta_pos / pos_delta_norm)
            new_position = current_position + pos_delta
            print(f"SAFETY: Limited position delta to {self.max_delta_pos}m")

        # Limit rotation delta
        rot_delta_norm = np.linalg.norm(rot_delta_rotvec)
        if rot_delta_norm > self.max_delta_rot:
            # Scale down the rotation delta and apply to current rotation
            rot_delta_scaled = rot_delta_rotvec * (self.max_delta_rot / rot_delta_norm)
            new_rotation = (
                R.from_rotvec(rot_delta_scaled).as_matrix() @ current_rotation
            )
            print(f"SAFETY: Limited rotation delta to {self.max_delta_rot}rad")

        return new_position, new_rotation

    def _calculate_action(self, poses):
        """Calculate action using delta-based transformation logic"""
        # Read robot observation
        robot_pos = poses["translation"]
        robot_rmat = poses["rotation"]

        # Initialize accumulated pose if not set
        if self.accumulated_pose is None:
            self.accumulated_pose = np.zeros(7)
            self.accumulated_pose[:3] = robot_pos
            self.accumulated_pose[3:7] = R.from_matrix(robot_rmat).as_quat()

        with self._state_lock:
            # Calculate gripper action based on trigger toggle
            trigger_value = self._state["buttons"].get(
                f"{self.controller_id.upper()}Tr", 0
            )
            trigger_pressed = trigger_value > 0.5

            # Toggle gripper state when trigger is pressed (rising edge detection)
            if trigger_pressed and not self.previous_trigger_pressed:
                self.gripper_state = 1 - self.gripper_state  # Toggle between 0 and 1
                print(f"Gripper: {'CLOSE' if self.gripper_state > 0 else 'OPEN'}")

            self.previous_trigger_pressed = trigger_pressed
            gripper = self.gripper_state  # Use the tracked gripper state

            # Handle grip button press for movement enabling
            grip_pressed = self._state["buttons"].get(
                f"{self.controller_id.upper()}G", False
            )

            # Check for A button press (recording toggle)
            a_button_pressed = self._state["buttons"].get("A", False)

            # Check for B button press (drop episode)
            b_button_pressed = self._state["buttons"].get("B", False)

            # Get current VR controller pose
            if self.controller_id not in self._state["poses"]:
                return {
                    "position": robot_pos,
                    "rmat": robot_rmat,
                    "gripper": gripper,
                    "a_button_pressed": a_button_pressed,
                    "b_button_pressed": b_button_pressed,
                }

            current_vr_transform = self._state["poses"][self.controller_id]
            current_vr_position, current_vr_rotation = self._extract_vr_pose(
                current_vr_transform
            )

            # Detect grip state transition and reset if grip is first pressed
            if grip_pressed and not self.previous_grip_state:
                # Reset state when grip is first pressed
                self.previous_vr_position = None
                self.previous_vr_rotation = None
                self.accumulated_pose = None

            self.previous_grip_state = grip_pressed

            # If movement is not enabled, return current pose
            if not grip_pressed or self.accumulated_pose is None:
                return {
                    "position": robot_pos,
                    "rmat": robot_rmat,
                    "gripper": gripper,
                    "a_button_pressed": a_button_pressed,
                    "b_button_pressed": b_button_pressed,
                }

            if self.previous_vr_position is None:
                # Initialize previous VR pose when grip is first pressed
                self.previous_vr_position = current_vr_position.copy()
                self.previous_vr_rotation = current_vr_rotation.copy()

            # delta calculation
            # Compute delta rotation and translation in VR controller frame
            delta_R_v = self.previous_vr_rotation.T @ current_vr_rotation
            delta_p_v = self.previous_vr_rotation.T @ (
                current_vr_position - self.previous_vr_position
            )

            # Apply scaling factors to delta translation and rotation
            delta_p_v_scaled = self.pos_scaling_factor * delta_p_v

            delta_R_v_scaled = R.from_matrix(delta_R_v).as_rotvec()
            delta_R_v_scaled = self.rot_scaling_factor * delta_R_v_scaled
            delta_R_v = R.from_rotvec(delta_R_v_scaled).as_matrix()

            # Transform delta rotation and scaled translation to robot's end-effector frame
            delta_R_e = self.R_ve @ delta_R_v @ self.R_ve.T
            delta_p_e = self.R_ve @ delta_p_v_scaled

            # Extract current accumulated pose
            current_position = self.accumulated_pose[:3]
            current_orientation_quat = self.accumulated_pose[3:7]
            current_rot = R.from_quat(current_orientation_quat).as_matrix()

            # Construct current transformation matrix T_e_t^b (end-effector pose in base frame)
            T_e_t_b = np.eye(4)
            T_e_t_b[:3, :3] = current_rot
            T_e_t_b[:3, 3] = current_position

            # Construct delta transformation matrix Delta_T_e^{e_t}
            Delta_T_e = np.eye(4)
            Delta_T_e[:3, :3] = delta_R_e
            Delta_T_e[:3, 3] = delta_p_e

            # Compute new end-effector pose in base frame
            # T_e_{t+1}^b = T_e_t^b * Delta_T_e^{e_t}
            T_e_new_b = np.dot(T_e_t_b, Delta_T_e)

            # Extract new position and orientation from T_e_new_b
            new_position = T_e_new_b[:3, 3]
            new_rot = T_e_new_b[:3, :3]
            new_orientation_quat = R.from_matrix(new_rot).as_quat()

            # Check safety limits before applying the movement
            if not self._check_safety_limits(
                new_position, new_rot, current_position, current_rot
            ):
                print("EMERGENCY STOP: Safety violation, maintaining current pose")
                return {
                    "position": current_position,
                    "rmat": current_rot,
                    "gripper": gripper,
                    "a_button_pressed": a_button_pressed,
                    "b_button_pressed": b_button_pressed,
                }

            # Update accumulated pose
            self.accumulated_pose[:3] = new_position
            self.accumulated_pose[3:7] = new_orientation_quat

            # Update previous VR controller pose for next iteration
            self.previous_vr_position = current_vr_position.copy()
            self.previous_vr_rotation = current_vr_rotation.copy()

            return {
                "position": new_position,
                "rmat": new_rot,
                "gripper": gripper,
                "a_button_pressed": a_button_pressed,
                "b_button_pressed": b_button_pressed,
            }

    def forward(self, poses):
        """Main forward function to get actions from VR input"""
        # Check if poses are available
        if self._state["poses"] == {}:
            action = None
            return action

        return self._calculate_action(poses)

    def get_info(self):
        """Get additional information about controller state"""
        with self._state_lock:
            return {
                "controller_on": len(self._state["poses"]) > 0,
                "success": self._state["buttons"].get("A", False),
                "failure": self._state["buttons"].get("B", False),
                "movement_enabled": self._state["buttons"].get(
                    f"{self.controller_id.upper()}G", False
                ),
            }
