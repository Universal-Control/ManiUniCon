"""Sensor implementations for data collection and replay."""

import importlib
import warnings

_SENSORS = {
    "ReplaySensor": "replay",
    "RealSenseSensor": "realsense",
    "ZedSensor": "zed",
}

__all__ = []

for sensor_name, module_name in _SENSORS.items():
    try:
        module = importlib.import_module(f".{module_name}", package=__name__)
        globals()[sensor_name] = getattr(module, sensor_name)
        __all__.append(sensor_name)
    except (ImportError, AttributeError) as e:
        warnings.warn(f"Failed to import {sensor_name}: {e}")
