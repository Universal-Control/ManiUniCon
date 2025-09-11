"""Policy implementations for robot control."""

import importlib
import warnings

_POLICIES = {
    "TorchModelPolicy": "torch_model",
    "SpaceMousePolicy": "spacemouse",
    "KeyboardPolicy": "keyboard",
    "QuestPolicy": "quest",
    "GelloPolicy": "gello",
}

__all__ = []

for policy_name, module_name in _POLICIES.items():
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
        globals()[policy_name] = getattr(module, policy_name)
        __all__.append(policy_name)
    except (ImportError, AttributeError) as e:
        warnings.warn(f"Failed to import {policy_name}: {e}")
