import torch, math
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


class Sort(Dataset):
     # Constructor
     def __init__(self, n, d):
         self.R = 100. if d % 2 == 1  else math.log(d)*10/(math.sqrt(2))
         normal = torch.distributions.Normal(100., 10.)
         self.x = normal.rsample( [n,d ]  )
         sorted_x, _ = torch.sort(self.x, dim=1, descending=True)
         b = 0 if d % 2 == 1  else -1
         w = torch.pow(-torch.ones(d), torch.arange(d)+2)
         f = (sorted_x @ w) - b
         self.y = 2*(torch.tensor(( f < self.R), dtype=float)).unsqueeze(1) - 1
         self.len = n
         self.f = f
         print( torch.sum((self.y + 1)/2)/n )
     # Getting the data
     def __getitem__(self, index):
         return self.x[index], self.y[index]
     # Getting length of the data
     def __len__(self):
         return self.len

class Ball(Dataset):
     # Constructor
     def __init__(self, n,d):
         self.R = 0.5*( math.sqrt(d) )
         beta = torch.distributions.Beta(torch.tensor([2.]), torch.tensor([2.]))
         normal = torch.distributions.Normal(0., 1.)
         self.x = beta.rsample([n]) *  normal.rsample([n,d])
         self.y = 2*(torch.tensor((torch.norm(self.x,dim=1) < self.R), dtype=float)).unsqueeze(1) - 1
         self.len = n
     # Getting the data
     def __getitem__(self, index):
         return self.x[index], self.y[index]
     # Getting length of the data
     def __len__(self):
         return self.len
