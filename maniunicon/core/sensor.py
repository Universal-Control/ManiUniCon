import torch.multiprocessing as mp

from maniunicon.utils.shared_memory.shared_storage import SharedStorage


class BaseSensor(mp.Process):
    """Base class for all sensors."""

    def __init__(
        self,
        shared_storage: SharedStorage,
        frequency: float = 100.0,  # Hz
        name: str = "Sensor",
    ):
        super().__init__(name=name)
        self.shared_storage = shared_storage
        self.frequency = frequency
        self.data_to_save = []
        self.record_dir = None

    def run(self):
        """Main process loop."""
        raise NotImplementedError("Subclasses must implement run")

    def stop(self):
        """Stop the sensor process."""
        self.shared_storage.is_running.value = False
        self.join()
