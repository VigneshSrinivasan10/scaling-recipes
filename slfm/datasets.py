import torch
from torch.utils.data import Dataset, DataLoader

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

    
class CustomDataset(Dataset):
    def __init__(self, size: int):
        self.data = DatasetCreator(size).create()
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
            return self.data[idx]

    def create(self, batch_size: int = 10000, shuffle: bool = True):
        return DataLoader(self.data, batch_size=batch_size, shuffle=shuffle)