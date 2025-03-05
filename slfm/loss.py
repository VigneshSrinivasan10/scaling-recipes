import torch

class ConditionalFlowMatchingLoss:
    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def __call__(self, flow_model, data):
        x = next(iter(data))
        t = torch.rand(x.shape[0], device=x.device)
        noise = torch.randn_like(x)

        x_t = (1 - (1 - self.sigma_min) * t[:, None]) * noise + t[:, None] * x
        optimal_flow = x - (1 - self.sigma_min) * noise
        predicted_flow = flow_model(x_t, time=t)

        return (predicted_flow - optimal_flow).square().mean()

class ClassificationLoss:
    def __init__(self, num_classes=10):
        self.num_classes = num_classes

    def __call__(self, model, x, y):
        y_hat = model(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss.mean()

