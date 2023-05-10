import torch, math
from torch.utils.data import Dataset
from torch_geometric.utils.random import erdos_renyi_graph
from torch_geometric.utils.sparse import to_torch_coo_tensor
from torch_geometric.data import Data, Batch
from torch_geometric.utils.convert import to_networkx
import networkx as nx

class Parity(Dataset):
    # Constructor
    def __init__(self, n, d):
        self.x = torch.randint(low=0, high=2, size=(n,d), dtype=torch.float32)
        self.y = torch.pow(-1,torch.sum(self.x, dim=1)).unsqueeze(1)
        print('Ratio  +/-:\t', torch.sum((self.y + 1)/2)/n )
        self.len = n
    # Getting the data
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

class Connectivity(Dataset):
    # Constructor
    def __init__(self, n, d):
        p = math.log(d)/d
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
            graphs.append( Data(edge_index=edge_index, x = torch.ones((d,1)), v=v  ) )
        self.y = torch.tensor(y).unsqueeze(1)
        print('Ratio  +/-:\t', torch.sum((self.y + 1)/2)/n )
        self.x = torch.cat( x, dim=0 )
        self.len = n
        self.graphs = graphs
        self.use_graphs = False
    # Getting the data
    def __getitem__(self, index):
        if self.use_graphs:
            return Batch().from_data_list( [ self.graphs[i] for i in index ] ), self.y[index]
        else:
            return self.x[index], self.y[index]
    # Getting length of the data
    def __len__(self):
        return self.len

class Sort(Dataset):
     # Constructor
     def __init__(self, n, d):
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
     # Getting the data
     def __getitem__(self, index):
         return self.x[index], self.y[index]
     # Getting length of the data
     def __len__(self):
         return self.len

class Ball(Dataset):
     # Constructor
     def __init__(self, n,d):
         self.R = 1.
         normal = torch.distributions.Normal(0., 1.)
         self.x = normal.rsample([n,d])
         self.x /= self.x.norm(dim=1, keepdim=True)
         normal = torch.distributions.Normal(0.,math.sqrt(-math.log(d)))
         self.x += normal.rsample([n,d])
         self.y = 2*(torch.tensor((torch.norm(self.x,dim=1) < self.R), dtype=float)).unsqueeze(1) - 1
         self.len = n
         print('Ratio  +/-:\t', torch.sum((self.y + 1)/2)/n )
     # Getting the data
     def __getitem__(self, index):
         return self.x[index], self.y[index]
     # Getting length of the data
     def __len__(self):
         return self.len
