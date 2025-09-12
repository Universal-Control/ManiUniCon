import numpy as np
from collections import deque
from typing import Optional, List, Tuple
import time


class LowPassFilter:
    def __init__(self, alpha, initial_value):
        self.alpha = alpha
        self.y = initial_value

    def filter(self, x):
        self.y = self.alpha * x + (1 - self.alpha) * self.y
        return self.y


class JointSpaceSmoother:
    """
    Online smoother for robot control in joint space.
    Filters out high-frequency noise and tremors from human hand motion
    while maintaining responsive control.
    """

    def __init__(
        self,
        num_joints: int,
        alpha_ewma: float = 0.3,
        window_size: int = 5,
        velocity_limit: Optional[np.ndarray] = None,
        acceleration_limit: Optional[np.ndarray] = None,
        deadband_threshold: float = 0.001,
        adaptive_alpha: bool = True,
    ):
        """
        Initialize the joint space smoother.

        Args:
            num_joints: Number of robot joints
            alpha_ewma: EWMA filter coefficient (0-1, higher = less smoothing)
            window_size: Size of moving average window
            velocity_limit: Max velocity for each joint (rad/s)
            acceleration_limit: Max acceleration for each joint (rad/s²)
            deadband_threshold: Minimum change to register movement
            adaptive_alpha: Enable adaptive smoothing based on motion dynamics
        """
        self.num_joints = num_joints
        self.alpha_ewma = alpha_ewma
        self.window_size = window_size
        self.deadband_threshold = deadband_threshold
        self.adaptive_alpha = adaptive_alpha

        # Initialize limits
        self.velocity_limit = (
            velocity_limit if velocity_limit is not None else np.ones(num_joints) * 2.0
        )
        self.acceleration_limit = (
            acceleration_limit
            if acceleration_limit is not None
            else np.ones(num_joints) * 5.0
        )

        # State variables
        self.position_history = deque(maxlen=window_size)
        self.filtered_position = None
        self.last_position = None
        self.last_velocity = np.zeros(num_joints)
        self.last_timestamp = None

        # Kalman filter states
        self.kf_state = np.zeros(num_joints * 2)  # [position, velocity] for each joint
        self.kf_covariance = np.eye(num_joints * 2) * 0.1

        # Statistics for adaptive filtering
        self.motion_variance = np.zeros(num_joints)
        self.variance_alpha = 0.1

    def smooth(
        self, joint_positions: np.ndarray, timestamp: Optional[float] = None
    ) -> np.ndarray:
        """
        Apply smoothing to joint positions.

        Args:
            joint_positions: Current joint positions (radians)
            timestamp: Current timestamp (seconds), if None uses time.time()

        Returns:
            Smoothed joint positions
        """
        if timestamp is None:
            timestamp = time.time()

        joint_positions = np.array(joint_positions)

        # Initialize on first call
        if self.filtered_position is None:
            self.filtered_position = joint_positions.copy()
            self.last_position = joint_positions.copy()
            self.last_timestamp = timestamp
            self.position_history.append(joint_positions)
            return joint_positions

        dt = (
            timestamp - self.last_timestamp if self.last_timestamp else 0.016
        )  # Default 60Hz

        # Apply multiple smoothing techniques
        smoothed = joint_positions.copy()

        # 1. Deadband filter - remove tiny tremors
        smoothed = self._apply_deadband(smoothed)

        # 2. Moving average filter
        smoothed = self._apply_moving_average(smoothed)

        # 3. Exponential weighted moving average
        smoothed = self._apply_ewma(smoothed)

        # 4. Velocity limiting
        smoothed = self._apply_velocity_limit(smoothed, dt)

        # 5. Acceleration limiting
        smoothed = self._apply_acceleration_limit(smoothed, dt)

        # 6. Kalman filtering for optimal estimation
        if dt > 0:
            smoothed = self._apply_kalman_filter(smoothed, dt)

        # Update state
        self.last_position = smoothed.copy()
        self.last_timestamp = timestamp
        self.position_history.append(joint_positions)
        self.filtered_position = smoothed

        # Update motion statistics for adaptive filtering
        self._update_motion_statistics(joint_positions)

        return smoothed

    def _apply_deadband(self, positions: np.ndarray) -> np.ndarray:
        """Apply deadband filtering to remove tiny movements."""
        if self.last_position is not None:
            delta = positions - self.last_position
            mask = np.abs(delta) < self.deadband_threshold
            positions[mask] = self.last_position[mask]
        return positions

    def _apply_moving_average(self, positions: np.ndarray) -> np.ndarray:
        """Apply moving average filter."""
        if len(self.position_history) > 0:
            history_array = np.array(self.position_history)
            weights = np.exp(np.linspace(-1, 0, len(history_array)))
            weights /= weights.sum()
            weighted_avg = np.average(history_array, axis=0, weights=weights)
            return 0.7 * positions + 0.3 * weighted_avg
        return positions

    def _apply_ewma(self, positions: np.ndarray) -> np.ndarray:
        """Apply exponential weighted moving average."""
        if self.filtered_position is not None:
            # Adaptive alpha based on motion variance
            if self.adaptive_alpha:
                alpha = self._compute_adaptive_alpha()
            else:
                alpha = self.alpha_ewma

            return alpha * positions + (1 - alpha) * self.filtered_position
        return positions

    def _compute_adaptive_alpha(self) -> np.ndarray:
        """Compute adaptive smoothing coefficient based on motion characteristics."""
        # Higher variance = less smoothing (higher alpha)
        # Lower variance = more smoothing (lower alpha)
        normalized_variance = np.clip(self.motion_variance / 0.01, 0, 1)
        alpha = self.alpha_ewma + (1 - self.alpha_ewma) * normalized_variance * 0.5
        return alpha

    def _apply_velocity_limit(self, positions: np.ndarray, dt: float) -> np.ndarray:
        """Apply velocity limiting to prevent sudden jumps."""
        if self.last_position is not None and dt > 0:
            velocity = (positions - self.last_position) / dt
            velocity_magnitude = np.abs(velocity)

            # Clip velocities that exceed limits
            scale_factors = np.minimum(
                1.0, self.velocity_limit / (velocity_magnitude + 1e-6)
            )
            limited_velocity = velocity * scale_factors

            return self.last_position + limited_velocity * dt
        return positions

    def _apply_acceleration_limit(self, positions: np.ndarray, dt: float) -> np.ndarray:
        """Apply acceleration limiting for smooth motion."""
        if self.last_position is not None and dt > 0:
            velocity = (positions - self.last_position) / dt
            acceleration = (velocity - self.last_velocity) / dt

            # Limit acceleration
            acc_magnitude = np.abs(acceleration)
            scale_factors = np.minimum(
                1.0, self.acceleration_limit / (acc_magnitude + 1e-6)
            )
            limited_acceleration = acceleration * scale_factors

            # Reconstruct position from limited acceleration
            new_velocity = self.last_velocity + limited_acceleration * dt
            new_position = self.last_position + new_velocity * dt

            self.last_velocity = new_velocity
            return new_position
        return positions

    def _apply_kalman_filter(self, positions: np.ndarray, dt: float) -> np.ndarray:
        """Apply Kalman filter for optimal state estimation."""
        # Simplified Kalman filter for each joint independently
        Q = np.eye(self.num_joints * 2) * 0.001  # Process noise
        R = np.eye(self.num_joints) * 0.01  # Measurement noise

        # State transition matrix
        F = np.eye(self.num_joints * 2)
        for i in range(self.num_joints):
            F[i * 2, i * 2 + 1] = dt  # Position += velocity * dt

        # Measurement matrix (observe positions only)
        H = np.zeros((self.num_joints, self.num_joints * 2))
        for i in range(self.num_joints):
            H[i, i * 2] = 1

        # Predict
        self.kf_state = F @ self.kf_state
        self.kf_covariance = F @ self.kf_covariance @ F.T + Q

        # Update
        y = positions - H @ self.kf_state  # Innovation
        S = H @ self.kf_covariance @ H.T + R  # Innovation covariance
        K = self.kf_covariance @ H.T @ np.linalg.inv(S)  # Kalman gain

        self.kf_state = self.kf_state + K @ y
        self.kf_covariance = (np.eye(self.num_joints * 2) - K @ H) @ self.kf_covariance

        # Extract filtered positions
        filtered_positions = np.array(
            [self.kf_state[i * 2] for i in range(self.num_joints)]
        )
        return filtered_positions

    def _update_motion_statistics(self, positions: np.ndarray):
        """Update motion variance for adaptive filtering."""
        if self.last_position is not None:
            instant_variance = np.var(positions - self.last_position)
            self.motion_variance = (
                self.variance_alpha * instant_variance
                + (1 - self.variance_alpha) * self.motion_variance
            )

    def reset(self):
        """Reset the smoother state."""
        self.position_history.clear()
        self.filtered_position = None
        self.last_position = None
        self.last_velocity = np.zeros(self.num_joints)
        self.last_timestamp = None
        self.kf_state = np.zeros(self.num_joints * 2)
        self.kf_covariance = np.eye(self.num_joints * 2) * 0.1
        self.motion_variance = np.zeros(self.num_joints)


class AdaptiveButterworth:
    """
    Adaptive Butterworth filter for frequency-domain smoothing.
    """

    def __init__(
        self, num_joints: int, cutoff_freq: float = 5.0, sample_rate: float = 60.0
    ):
        """
        Initialize Butterworth filter.

        Args:
            num_joints: Number of joints
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sampling rate in Hz
        """
        self.num_joints = num_joints
        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate

        # Filter coefficients
        self.omega = 2 * np.pi * cutoff_freq / sample_rate
        self.alpha = np.sin(self.omega) / np.sqrt(2)

        # Filter states
        self.x_prev = np.zeros((2, num_joints))
        self.y_prev = np.zeros((2, num_joints))
        self.initialized = False

    def filter(self, positions: np.ndarray) -> np.ndarray:
        """Apply Butterworth filtering."""
        if not self.initialized:
            self.x_prev[0] = positions
            self.x_prev[1] = positions
            self.y_prev[0] = positions
            self.y_prev[1] = positions
            self.initialized = True
            return positions

        # Second-order Butterworth filter
        a0 = 1 + self.alpha
        a1 = -2 * np.cos(self.omega) / a0
        a2 = (1 - self.alpha) / a0
        b0 = (1 - np.cos(self.omega)) / (2 * a0)
        b1 = (1 - np.cos(self.omega)) / a0
        b2 = b0

        # Apply filter
        y = b0 * positions + b1 * self.x_prev[0] + b2 * self.x_prev[1]
        y -= a1 * self.y_prev[0] + a2 * self.y_prev[1]

        # Update states
        self.x_prev[1] = self.x_prev[0]
        self.x_prev[0] = positions
        self.y_prev[1] = self.y_prev[0]
        self.y_prev[0] = y

        return y


# Example usage
if __name__ == "__main__":
    # Create smoother for 6-DOF robot
    smoother = JointSpaceSmoother(
        num_joints=7,
        alpha_ewma=0.3,
        window_size=5,
        velocity_limit=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        acceleration_limit=np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
        deadband_threshold=0.001,
        adaptive_alpha=True,
    )

    # Optional: Add Butterworth filter for additional frequency-domain smoothing
    butterworth = AdaptiveButterworth(num_joints=7, cutoff_freq=5.0, sample_rate=100.0)

    # Simulate noisy teleoperation input
    np.random.seed(42)
    time_steps = 100
    t = np.linspace(0, 5, time_steps)

    # Generate smooth trajectory with added noise (simulating hand tremor)
    true_positions = np.array([np.sin(t * 2) for _ in range(7)]).T
    noise = np.random.normal(0, 0.05, true_positions.shape)
    tremor = 0.02 * np.sin(t * 50).reshape(-1, 1)  # High-frequency tremor
    noisy_positions = true_positions + noise + tremor

    # Apply smoothing
    smoothed_positions = []
    for i, pos in enumerate(noisy_positions):
        start_time = time.perf_counter()
        pos = smoother.smooth(pos, timestamp=t[i])
        smooth_pos = butterworth.filter(pos)  # Optional second stage
        end_time = time.perf_counter()
        print(f"Time taken: {end_time - start_time:.6f} seconds")
        smoothed_positions.append(smooth_pos)

    smoothed_positions = np.array(smoothed_positions)

    print(
        f"Input noise RMS: {np.sqrt(np.mean((noisy_positions - true_positions)**2)):.4f}"
    )
    print(
        f"Smoothed error RMS: {np.sqrt(np.mean((smoothed_positions - true_positions)**2)):.4f}"
    )
    print(
        f"Smoothing improvement: {(1 - np.sqrt(np.mean((smoothed_positions - true_positions)**2)) / np.sqrt(np.mean((noisy_positions - true_positions)**2))) * 100:.1f}%"
    )
