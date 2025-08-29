import torch
import hydra


class PPTPolicyModel:
    def __init__(
        self, domain, network, head, stem, pretrained_model_path, device="cpu"
    ):
        self.domain = domain
        self.model = hydra.utils.instantiate(network)
        self.model.init_domain_stem(domain, stem)
        self.model.init_domain_head(domain, head)

        self.model.finalize_modules()

        if len(pretrained_model_path):
            print("=" * 80)
            print(f"Loading pretrained model from {pretrained_model_path}")
            print("=" * 80)
            self.model.load_state_dict(torch.load(pretrained_model_path))
        self.model.to(device)

        n_parameters = sum(p.numel() for p in self.model.parameters())
        print(f"number of params (M): {n_parameters / 1.0e6:.2f}")

        self.model.eval()

    def __call__(self, obs, **kwargs):
        """
        Forward pass through the model.
        :param obs: Observation input to the model.
        :return: Model output.
        """
        with torch.no_grad():
            return self.model(domain=self.domain, data=obs, **kwargs)
