from typing import List, Tuple, Optional, Dict, Union, Any
import os
import math
import numpy as np


def get_accumulate_timestamp_idxs(
    timestamps: List[float] | float,
    start_time: float,
    dt: float,
    eps: float = 1e-5,
    next_global_idx: Optional[int] = 0,
    allow_negative: bool = False,
) -> Tuple[List[int], List[int], int]:
    """
    For each dt window, choose the first timestamp in the window.
    Assumes timestamps sorted. One timestamp might be chosen multiple times due to dropped frames.
    next_global_idx should start at 0 normally, and then use the returned next_global_idx.
    However, when overwiting previous values are desired, set last_global_idx to None.

    Returns:
    local_idxs: which index in the given timestamps array to chose from
    global_idxs: the global index of each chosen timestamp
    next_global_idx: used for next call.
    """

    if isinstance(timestamps, float):
        timestamps = [timestamps]

    local_idxs = list()
    global_idxs = list()
    for local_idx, ts in enumerate(timestamps):
        # add eps * dt to timestamps so that when ts == start_time + k * dt
        # is always recorded as kth element (avoiding floating point errors)
        global_idx = math.floor((ts - start_time) / dt + eps)
        if (not allow_negative) and (global_idx < 0):
            continue
        if next_global_idx is None:
            next_global_idx = global_idx

        n_repeats = max(0, global_idx - next_global_idx + 1)
        for i in range(n_repeats):
            local_idxs.append(local_idx)
            global_idxs.append(next_global_idx + i)
        next_global_idx += n_repeats
    return local_idxs, global_idxs, next_global_idx


def align_timestamps(
    timestamps: List[float],
    target_global_idxs: List[int],
    start_time: float,
    dt: float,
    eps: float = 1e-5,
):
    if isinstance(target_global_idxs, np.ndarray):
        target_global_idxs = target_global_idxs.tolist()
    assert len(target_global_idxs) > 0

    local_idxs, global_idxs, _ = get_accumulate_timestamp_idxs(
        timestamps=timestamps,
        start_time=start_time,
        dt=dt,
        eps=eps,
        next_global_idx=target_global_idxs[0],
        allow_negative=True,
    )
    if len(global_idxs) > len(target_global_idxs):
        # if more steps available, truncate
        global_idxs = global_idxs[: len(target_global_idxs)]
        local_idxs = local_idxs[: len(target_global_idxs)]

    if len(global_idxs) == 0:
        import pdb

        pdb.set_trace()

    for i in range(len(target_global_idxs) - len(global_idxs)):
        # if missing, repeat
        local_idxs.append(len(timestamps) - 1)
        global_idxs.append(global_idxs[-1] + 1)
    assert global_idxs == target_global_idxs
    assert len(local_idxs) == len(global_idxs)
    return local_idxs


class TimestampAlignedBuffer:
    def __init__(
        self,
        start_time: float,
        dt: float,
        max_record_steps: int,
        eps: float = 1e-5,
        overwrite: bool = False,
    ):
        self.start_time = start_time
        self.dt = dt
        self.eps = eps
        self.max_record_steps = max_record_steps

        self.data_buffer = None
        self.timestamp_buffer = None
        self.size = 0  # Track actual used size
        self.recording_stopped = (
            False  # Flag to track if recording was stopped due to limit
        )
        # action will overwrite previous values
        # while obs will not
        self.overwrite = overwrite
        if overwrite:
            self.next_global_idx = None
        else:
            self.next_global_idx = 0

    def __len__(self):
        if self.overwrite:
            return self.size
        else:
            return self.next_global_idx

    def add(
        self,
        data: Dict[str, Union[np.ndarray, float, int, str, Any]],
        timestamp: float,
    ):
        # Check if recording should be stopped
        if self.recording_stopped:
            return

        _, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(
            timestamps=timestamp,
            start_time=self.start_time,
            dt=self.dt,
            eps=self.eps,
            next_global_idx=self.next_global_idx,
        )

        if not self.overwrite:
            self.next_global_idx = next_global_idx

        if len(global_idxs) > 0:
            # Check if we would exceed the maximum record steps
            max_required_size = global_idxs[-1] + 1
            if max_required_size > self.max_record_steps:
                print(
                    f"[TimestampAlignedBuffer] Recording stopped: reached maximum record steps ({self.max_record_steps})"
                )
                self.recording_stopped = True
                return

            if self.timestamp_buffer is None:
                # Pre-allocate buffers based on max_record_steps
                self.data_buffer = dict()
                for key, value in data.items():
                    if isinstance(value, np.ndarray):
                        # Handle numpy arrays
                        buffer_shape = (self.max_record_steps,) + value.shape
                        self.data_buffer[key] = np.zeros(
                            buffer_shape, dtype=value.dtype
                        )
                    else:
                        # Handle scalars (float, int, str, etc.)
                        if isinstance(value, str):
                            # For strings, create object array to handle variable length strings
                            self.data_buffer[key] = np.empty(
                                self.max_record_steps, dtype=object
                            )
                        else:
                            # For numeric scalars (float, int, etc.)
                            self.data_buffer[key] = np.zeros(
                                self.max_record_steps, dtype=type(value)
                            )

                self.timestamp_buffer = np.zeros(
                    self.max_record_steps, dtype=np.float64
                )

            # Write data to buffers using global indices
            for key, value in data.items():
                if key in self.data_buffer:
                    self.data_buffer[key][global_idxs] = value

            # Update timestamp buffer
            self.timestamp_buffer[global_idxs] = timestamp

            # Update size tracking
            if self.overwrite:
                self.size = max(self.size, max_required_size)
            else:
                self.size = max_required_size

    @property
    def data(self):
        """Return the actual data up to the used size"""
        if self.timestamp_buffer is None:
            return dict()
        result = dict()
        for key, value in self.data_buffer.items():
            result[key] = value[: self.size]
        return result

    @property
    def timestamps(self):
        """Return the actual timestamps up to the used size"""
        if self.timestamp_buffer is None:
            return np.array([])
        return self.timestamp_buffer[: self.size]

    def dump(self, name: str, dir: str):
        np.savez(
            os.path.join(dir, f"{name}.npz"),
            _buffer_timestamps=self.timestamps,
            **self.data,
        )
