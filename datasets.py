import torch, math
from torch.utils.data import Dataset
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.utils.sparse import to_torch_coo_tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import to_networkx
import networkx as nx

class Parity(Dataset):
    # Constructor
    def __init__(self, n, d, log=False):
        self.x = torch.randint(low=0, high=2, size=(n,d), dtype=torch.float32)
        self.y = torch.pow(-1,torch.sum(self.x, dim=1)).unsqueeze(1)
        print('Ratio  +/-:\t', torch.sum((self.y + 1)/2)/n )
        if log: print('Constant classifier:', max( [ 1 - (torch.sum((self.y + 1)/2)/n).item(), (torch.sum((self.y + 1)/2)/n).item()  ]  ) )
        self.len = n
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

class Connectivity(Dataset):
    # Constructor
    def __init__(self, n, d, log=False):
        p = (1.1*math.log(d))/d
        graphs = []
        y = []
        x = []
        for i in range(n):
            edge_index = erdos_renyi_graph(d,p)
            v = to_torch_coo_tensor(edge_index, size=d).to_dense().view((1,d*d))
            G = to_networkx( Data(edge_index=edge_index, x = torch.ones((d,1)), v=v  ), to_undirected=True  )
            if nx.is_connected(G):
                y.append(1)
            else:
                y.append(-1)
            x.append(v)
            graphs.append( to_torch_coo_tensor(edge_index, size=d) )
        self.y = torch.tensor(y).unsqueeze(1).float()
        print('Ratio  +/-:\t', torch.sum((self.y + 1)/2)/n )
        if log: print('Constant classifier:', max( [ 1 - (torch.sum((self.y + 1)/2)/n).item(), (torch.sum((self.y + 1)/2)/n).item()  ]  ) )
        self.x = torch.cat( x, dim=0 )
        self.len = n
        self.graphs = graphs
        self.use_graphs = False
    # Getting the data
    def __getitem__(self, index):
        if self.use_graphs:
            return self.graphs[index], self.y[index]
        else:
            return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

class Sort(Dataset):
     # Constructor
     def __init__(self, n, d, log=False):
         mean = 0.
         scale = 1.
         normal = torch.distributions.Normal(mean, scale)

         b = 0 if d % 2 == 1  else -1
         w = torch.pow(-torch.ones(d), torch.arange(d)+2)

         self.R = 0 if d%2 == 1 else round(math.log(d)/2) + math.sqrt(2)
         if self.R > 0 and d >= 9000: self.R -= math.sqrt(2)

         self.x = normal.rsample( [n,d ]  )
         sorted_x, _ = torch.sort(self.x, dim=1, descending=True)
         f = (sorted_x @ w) - b

         self.y = 2*(torch.tensor(( f < self.R), dtype=float)).unsqueeze(1) - 1
         self.len = n
         self.f = f
         print('Ratio  +/-:\t', torch.sum((self.y + 1)/2)/n )
         if log: print('Constant classifier:', max( [ 1 - (torch.sum((self.y + 1)/2)/n).item(), (torch.sum((self.y + 1)/2)/n).item()  ]  ) )
         # Getting the data
     def __getitem__(self, index):
         return self.x[index], self.y[index]
     # Getting length of the data
     def __len__(self):
         return self.len

class Range(Dataset):
      # Constructor
      def __init__(self, n, d, log=False):
          self.x = torch.randint(10, (n, d)).float()
          f = []
          for row in self.x:
              f.append(len(torch.unique(row)))
          f = torch.tensor(f)
          self.y = 2*(torch.tensor(( f < 816), dtype=float)).unsqueeze(1) - 1
          self.len = n
          self.f = f
          print('Ratio  +/-:\t', torch.sum((self.y + 1)/2)/n )
          if log: print('Constant classifier:', max( [ 1 - (torch.sum((self.y + 1)/2)/n).item(), (torch.sum((self.y + 1)/2)/n).item()  ]  ) )
          # Getting the data
      def __getitem__(self, index):
          return self.x[index], self.y[index]
      # Getting length of the data
      def __len__(self):
          return self.len


class Ball(Dataset):
     # Constructor
     def __init__(self, n,d,log=False):
         self.R = 1.
         normal = torch.distributions.Normal(0., 1.)
         self.x = normal.rsample([n,d])
         self.x /= self.x.norm(dim=1, keepdim=True)
         scaling_factor = .1
         eps = torch.randn(n) * scaling_factor
         norms = torch.norm(self.x, dim=1)
         new_norms = norms + eps
         self.x = self.x * (new_norms / norms).unsqueeze(1)
         self.y = 2*(torch.tensor((torch.norm(self.x,dim=1) < self.R), dtype=float)).unsqueeze(1) - 1
         self.len = n
         print('Ratio  +/-:\t', torch.sum((self.y + 1)/2)/n )
         if log: print('Constant classifier:', max( [ 1 - (torch.sum((self.y + 1)/2)/n).item(), (torch.sum((self.y + 1)/2)/n).item()  ]  ) )
     # Getting the data
     def __getitem__(self, index):
         return self.x[index], self.y[index]
     # Getting length of the data
     def __len__(self):
         return self.len

def dot( x, a ):
     return torch.bmm(x.view(x.shape[0], 1, x.shape[1]), a.view(a.shape[0], a.shape[1], 1)).reshape(x.shape[0],1)

class Sin(Dataset):
     # Constructor
     def __init__(self, n,d,log=False):
         normal = torch.distributions.Normal(0., 1.)
         self.x = normal.rsample([n,d//2])
         self.x /= self.x.norm(dim=1, keepdim=True)
         self.x_ = normal.rsample([n,d//2])
         self.x_ /= self.x_.norm(dim=1, keepdim=True)
         self.f = torch.sin( math.pi * math.pow(d//2+1,3) * dot( self.x, self.x_ ) )
         self.x = torch.cat([self.x, self.x_], dim=1)
         self.y = 2*(torch.tensor((self.f < 0), dtype=float)) - 1
         print(self.x.shape, self.y.shape)
         self.len = n
         print('Ratio  +/-:\t', torch.sum((self.y + 1)/2)/n )
         if log: print('Constant classifier:', max( [ 1 - (torch.sum((self.y + 1)/2)/n).item(), (torch.sum((self.y + 1)/2)/n).item()  ]  ) )
     # Getting the data
     def __getitem__(self, index):
         return self.x[index], self.y[index]
     # Getting length of the data
     def __len__(self):
         return self.len
