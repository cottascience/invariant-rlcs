import torch
from torch.utils.data import Dataset

class Parity(Dataset):
    # Constructor
    def __init__(self, n, d):
        self.x = x = torch.randint(low=0, high=2, size=(n,d), dtype=torch.float32)
        self.y = torch.pow(-1,torch.sum(x, dim=1))
        self.len = n
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len
