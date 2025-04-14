import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

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
    
class MNISTDataset(Dataset):
    def __init__(self, size: int = 50000, batch_size: int = 100, shuffle: bool = True):
        self.dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
        self.data = self.dataset.data[:size]
        self.data = self.data.view(-1, 28*28)
        self.data = self.data.float() / 255.0
        self.labels = self.dataset.targets[:size]
        
        self.val_data = self.dataset.data[size:]
        self.val_data = self.val_data.view(-1, 28*28)
        self.val_data = self.val_data.float() / 255.0
        self.val_labels = self.dataset.targets[size:]

        self.test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
        self.test_data = self.test_dataset.data[:size]
        self.test_data = self.test_data.view(-1, 28*28)
        self.test_data = self.test_data.float() / 255.0
        self.test_labels = self.test_dataset.targets[:size]
        
        self.batch_size = batch_size
        self.shuffle = shuffle
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def create(self, type: str = "train"):
        if type == "train":
            dataset = torch.utils.data.TensorDataset(self.data, self.labels)
        elif type == "val":
            dataset = torch.utils.data.TensorDataset(self.val_data, self.val_labels)
        elif type == "test":
            dataset = torch.utils.data.TensorDataset(self.test_data, self.test_labels)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=self.shuffle)

