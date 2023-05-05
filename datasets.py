import torch
from torch.utils.data import Dataset

class Parity(Dataset):
    # Constructor
    def __init__(self, n, d):
        self.x = torch.randint(low=0, high=2, size=(n,d), dtype=torch.float32)
        self.y = torch.pow(-1,torch.sum(self.x, dim=1)).unsqueeze(1)
        self.len = n
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

class Ball(Dataset):
     # Constructor
     def __init__(self, n, d, R=0.5, L=1):
         self.x = (torch.rand(n, d) - 0.5) * L
         self.y = torch.pow(torch.tensor((torch.norm(self.x,dim=1) < R), dtype=float))
         self.len = n
     # Getting the data
     def __getitem__(self, index):
         return self.x[index], self.y[index]
     # Getting length of the data
     def __len__(self):
         return self.len
