from multiprocessing.synchronize import Event
import torch.multiprocessing as mp

from maniunicon.utils.shared_memory.shared_storage import SharedStorage


class BasePolicy(mp.Process):
    """Process of robot policy model."""

    def __init__(
        self,
        shared_storage: SharedStorage,
        reset_event: Event,
        command_latency: float = 0.01,  # seconds
        name: str = "Policy",
    ):
        super().__init__(name=name)
        self.shared_storage = shared_storage
        self.reset_event = reset_event
        self.command_latency = command_latency

    def run(self):
        """Main process loop."""
        raise NotImplementedError("Subclasses must implement run")

    def stop(self):
        """Stop the policy process."""
        self.shared_storage.is_running.value = False
        self.join()
