import torch

class DatasetCreator:
    def __init__(self, size: int):
        self.size = size
    def create(self) -> torch.Tensor:
        complex_points = torch.polar(torch.tensor(1.0), torch.rand(self.size) * 2 * torch.pi)
        X = torch.stack((complex_points.real, complex_points.imag)).T
        upper = complex_points.imag > 0
        left = complex_points.real < 0
        X[upper, 1] = 0.5
        X[upper & left, 0] = -0.5
        X[upper & ~left, 0] = 0.5
        noise = torch.zeros_like(X)
        noise[upper] = torch.randn_like(noise[upper]) * 0.10
        noise[~upper] = torch.randn_like(noise[~upper]) * 0.05
        X += noise
        X -= X.mean(axis=0)
        X /= X.std(axis=0)
        return X + noise

    