import torch

class ConditionalFlowMatchingLoss:
    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def __call__(self, flow_model, x):
        t = torch.rand(x.shape[0], device=x.device)
        noise = torch.randn_like(x)

        x_t = (1 - (1 - self.sigma_min) * t[:, None]) * noise + t[:, None] * x
        optimal_flow = x - (1 - self.sigma_min) * noise
        predicted_flow = flow_model(x_t, time=t)

        return (predicted_flow - optimal_flow).square().mean()
