import importlib
from collections import OrderedDict
from typing import Union

import numpy as np
import torch

TensorData = Union[np.ndarray, torch.Tensor]


def convert_to_torch(
    array: TensorData,
    dtype: torch.dtype = None,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Converts a given array into a torch tensor.

    The function tries to convert the array to a torch tensor. If the array is a numpy/warp arrays, or python
    list/tuples, it is converted to a torch tensor. If the array is already a torch tensor, it is returned
    directly.

    If ``device`` is None, then the function deduces the current device of the data. For numpy arrays,
    this defaults to "cpu", for torch tensors it is "cpu" or "cuda", and for warp arrays it is "cuda".

    Note:
        Since PyTorch does not support unsigned integer types, unsigned integer arrays are converted to
        signed integer arrays. This is done by casting the array to the corresponding signed integer type.

    Args:
        array: The input array. It can be a numpy array, warp array, python list/tuple, or torch tensor.
        dtype: Target data-type for the tensor.
        device: The target device for the tensor. Defaults to None.

    Returns:
        The converted array as torch tensor.
    """
    # Convert array to tensor
    # if the datatype is not currently supported by torch we need to improvise
    # supported types are: https://pytorch.org/docs/stable/tensors.html
    if isinstance(array, torch.Tensor):
        tensor = array
    elif isinstance(array, np.ndarray):
        if array.dtype == np.uint32:
            array = array.astype(np.int32)
        # need to deal with object arrays (np.void) separately
        tensor = torch.from_numpy(array)
    else:
        tensor = torch.Tensor(array)
    # Convert tensor to the right device
    if device is not None and str(tensor.device) != str(device):
        tensor = tensor.to(device)
    # Convert dtype of tensor if requested
    if dtype is not None and tensor.dtype != dtype:
        tensor = tensor.type(dtype)

    return tensor


def instantiate_class(class_path, *args, **kwargs):
    """
    Instantiates a class given its path as a string using importlib.

    Args:
        class_path (str): The path to the class (e.g., 'module.submodule.ClassName').
        *args: Positional arguments to pass to the class constructor.
        **kwargs: Keyword arguments to pass to the class constructor.

    Returns:
        An instance of the class.
    """
    parts = class_path.split(".")
    module_name = ".".join(parts[:-1])
    class_name = parts[-1]

    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)

    return cls(*args, **kwargs)


def dict_apply(x, func):
    dict_type = type(x)
    if type(x) is not dict_type:
        return func(x)

    result = dict_type()
    for key, value in x.items():
        if isinstance(value, (str, list)):
            result[key] = value
        elif isinstance(value, (dict_type, dict, OrderedDict)):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def temporal_depth_proc_batch(cur, prev, sigma_depth=0.02, sigma_rgb=0.01, eps=1e-6):
    """
    Batched temporal depth refinement.

    Parameters
    ----------
    cur, prev : Tuple[np.ndarray, np.ndarray]
        cur/prev_rgb : (B, H, W, 3), uint8 or float in [0,255]
        cur/prev_depth : (B, H, W), float32
    sigma_depth : float
        Base depth noise (meters) for all pixels.
    sigma_rgb : float
        Stabiliser to avoid div‑by‑zero in RGB ratio.
    eps : float
        Numerical epsilon to avoid zero division.
    Returns
    -------
    np.ndarray
        Refined depth, shape (B, H, W)
    """
    cur_rgb, cur_depth = cur
    prev_rgb, prev_depth = prev

    cur_depth = cur_depth[:, 0]
    prev_depth = prev_depth[:, 0]

    # (B, H, W)   broadcasted in later ops
    base_sigma_depth = sigma_depth * np.ones_like(cur_depth)

    # ∥ΔRGB∥∞  (B, H, W)
    diff_rgb = (
        np.abs(cur_rgb.astype(np.float32) - prev_rgb.astype(np.float32)).max(axis=-1)
    ) / 255.0
    ratio_rgb = (diff_rgb + sigma_rgb) / (diff_rgb + eps)  # avoid zero‑division
    ratio_rgb = ratio_rgb.clip(1.0, 5.0)

    _sigma_depth = base_sigma_depth * ratio_rgb  # (B, H, W)

    # |Δdepth|  (B, H, W)
    diff_depth = np.abs(cur_depth - prev_depth)  # (B, H, W)

    # compute mask ratio for current depth
    mask_ratio = diff_depth / (diff_depth + _sigma_depth + eps)  # (B, H, W)

    # merge depths
    refined_depth = cur_depth * mask_ratio + prev_depth * (1.0 - mask_ratio)

    return refined_depth.reshape(-1, 1, *refined_depth.shape[1:])  # (B, 1, H, W)
