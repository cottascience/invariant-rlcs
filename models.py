import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import MLP as torch_geometric_MLP
import math

def dot( x, a ):
    return torch.bmm(x.view(x.shape[0], 1, x.shape[1]), a.view(a.shape[0], a.shape[1], 1)).reshape(x.shape[0],1)

class DeepSets(torch.nn.Module):
      def __init__(self, hidden_size, num_layers, dropout_p, use_batchnorm):
          super(DeepSets, self).__init__()

          norm = 'batch_norm' if use_batchnorm else None
          act = nn.ReLU()
          self.phi = torch_geometric_MLP(in_channels = 1, hidden_channels = hidden_size, out_channels = hidden_size,
                         num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
          self.rho = torch_geometric_MLP(in_channels = hidden_size, hidden_channels = hidden_size, out_channels = 1,
                          num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
      def forward(self, x):
          out = 0
          for i in range(x.shape[1]):
              out += self.phi(x[:,i].view(x.shape[0],1))
          return torch.tanh(self.rho(out))

class RLC(torch.nn.Module):
     def __init__(self, noise_size, hidden_size, num_layers, dropout_p, use_batchnorm, x_size):
         super(RLC, self).__init__()

         norm = 'batch_norm' if use_batchnorm else None
         act = nn.ReLU()
         self.b = torch_geometric_MLP(in_channels = 2*noise_size, hidden_channels = hidden_size, out_channels = 1,
                        num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
         self.a = torch_geometric_MLP(in_channels = 2*noise_size, hidden_channels = hidden_size, out_channels = x_size,
                        num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
         self.ab = torch_geometric_MLP(in_channels = noise_size, hidden_channels = hidden_size, out_channels = 1+x_size,
                        num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
         self.noise_size = noise_size
         self.noise_dist = torch.distributions.Normal(0,1)
         self.c1 = torch.nn.Parameter(torch.ones(1)*1.)
         self.c2 = torch.nn.Parameter(torch.ones(1)*.1)
         self.layer_norm = nn.LayerNorm(1+x_size)
     def forward(self, x):
        noise = self.noise_dist.rsample([x.shape[0], self.noise_size]).to(x.device)
        ua = self.noise_dist.rsample([x.shape[0], self.noise_size]).to(x.device)
        ub = self.noise_dist.rsample([x.shape[0], self.noise_size]).to(x.device)
        #a = self.a(torch.cat([noise,ua],dim=1))
        #b = self.b(torch.cat([noise,ub],dim=1))
        ab = self.layer_norm(self.ab(noise))
        a = ab[:,:-1]
        b = ab[:,-1].unsqueeze(1)
        res = dot(x,a) - b
        return torch.tanh(res)

class RSetC(torch.nn.Module):
     def __init__(self, noise_size, hidden_size, num_layers, dropout_p, use_batchnorm, x_size):
         super(RSetC, self).__init__()

         norm = 'batch_norm' if use_batchnorm else None
         act = nn.LeakyReLU()
         self.a_mlp = torch_geometric_MLP(in_channels = 2*noise_size, hidden_channels = hidden_size, out_channels = 1,
                        num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
         self.b_mlp = torch_geometric_MLP(in_channels = 2*noise_size, hidden_channels = hidden_size, out_channels = 1,
                        num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
         self.noise_size = noise_size
         self.noise_dist = torch.distributions.Normal(0,1)
         self.c1 = torch.nn.Parameter(torch.ones(1)*1)
         self.c2 = torch.nn.Parameter(torch.ones(1)*1)

     def forward(self, x):
        u = self.noise_dist.rsample([x.shape[0], self.noise_size]).to(x.device)
        ub = self.noise_dist.rsample([x.shape[0], self.noise_size]).to(x.device)
        b = self.b_mlp( torch.cat([u,ub],dim=1) )
        a = []
        for i in range(x.shape[1]):
            ua = self.noise_dist.rsample([x.shape[0], self.noise_size]).to(x.device)
            a.append( self.a_mlp( torch.cat([u,ua],dim=1) ) )
        a = torch.cat(a, dim=1)
        res = dot(x,self.c1*a) - self.c2*b
        return torch.tanh(res)

class RGraphC(torch.nn.Module):
      def __init__(self, hidden_size, num_layers, dropout_p, use_batchnorm, x_size):
          super(RGraphC, self).__init__()

          norm = 'batch_norm' if use_batchnorm else None
          act = nn.ReLU()
          self.a_mlp = torch_geometric_MLP(in_channels = 4, hidden_channels = hidden_size, out_channels = 1,
                         num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
          self.b_mlp = torch_geometric_MLP(in_channels = 2, hidden_channels = 2*hidden_size, out_channels = 1,
                         num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
          self.noise_dist = torch.distributions.Normal(0,1)
          self.c1 = torch.nn.Parameter(torch.ones(1))
          self.c2 = torch.nn.Parameter(torch.ones(1))

      def make_undirected(self, X, N):
          B = X.shape[0]
          X = X.view(B, N, N)
          mask = torch.triu(torch.ones(N, N)) > 0
          upper_triangular = X[:, mask]
          num_selected = (N*(N+1)) // 2
          selected_X = upper_triangular.view(B, num_selected)
          return selected_X

      def forward(self, x, undirected=True):
         num_nodes = int(math.sqrt(x.shape[1]))
         u = self.noise_dist.rsample([x.shape[0], 1]).to(x.device)
         ub = self.noise_dist.rsample([x.shape[0], 1]).to(x.device)
         b = self.b_mlp( torch.cat([u,ub],dim=1) )
         uij = self.noise_dist.rsample([x.shape[0], num_nodes*num_nodes]).to(x.device)
         ui = self.noise_dist.rsample([x.shape[0], num_nodes]).to(x.device)
         uj = self.noise_dist.rsample([x.shape[0], num_nodes]).to(x.device)
         ui = ui.repeat_interleave(num_nodes).view(x.shape)
         uj = uj.t().repeat(num_nodes,1).t()

         if undirected:
             x = self.make_undirected(x,num_nodes)
             uij = self.make_undirected(uij, num_nodes)
             ui = self.make_undirected(ui, num_nodes)
             uj = self.make_undirected(uj, num_nodes)

         #uij = torch.ones_like(uij)
         #a = []
         #for i in range(x.shape[1]):
             #a.append( self.a_mlp( torch.cat([ u, uij[:,i].unsqueeze(1), ui[:,i].unsqueeze(1), uj[:,i].unsqueeze(1) ],dim=1) ) )
         #a = torch.cat(a, dim=1)

         u = u.repeat_interleave(x.shape[1],1).view(x.shape)
         U = torch.stack([u, uij, ui, uj], dim=2)
         U = U.view(x.shape[0]*x.shape[1], 4)
         a = self.a_mlp(U)
         a = a.view(x.shape)

         a = a.sigmoid()

         res = dot(x,self.c1*a) - self.c2*b

         return torch.tanh(res)

class RSphereC(torch.nn.Module):
      def __init__(self, noise_size, hidden_size, num_layers, dropout_p, use_batchnorm, x_size):
          super(RSphereC, self).__init__()

          norm = 'batch_norm' if use_batchnorm else None
          act = nn.LeakyReLU()
          self.sigma_mlp = torch_geometric_MLP(in_channels = 2*noise_size, hidden_channels = hidden_size, out_channels = 1,
                         num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
          self.b_mlp = torch_geometric_MLP(in_channels = 2*noise_size, hidden_channels = hidden_size, out_channels = 1,
                          num_layers=num_layers, norm=norm, dropout=dropout_p, act=act)
          self.noise_size = noise_size
          self.noise_dist = torch.distributions.Normal(0,1)
          self.c1 = torch.nn.Parameter(torch.ones(1))
          self.c2 = torch.nn.Parameter(torch.ones(1))
          self.normal = torch.distributions.Normal(0,1)
      def forward(self, x):
         u = self.noise_dist.rsample([x.shape[0], self.noise_size]).to(x.device)
         u_sigma = self.noise_dist.rsample([x.shape[0], self.noise_size]).to(x.device)
         u_b = self.noise_dist.rsample([x.shape[0], self.noise_size]).to(x.device)
         sigma = self.sigma_mlp( torch.cat([u,u_sigma],dim=1 ))
         b = self.b_mlp( torch.cat([u,u_b],dim=1))
         a = sigma * self.normal.rsample( x.shape  ).to(x.device)
         return F.tanh(dot(x,self.c1*a) - self.c2*b)

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GIN, self).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch_geometric_MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = torch_geometric_MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, B):
        x, edge_index, batch = B.x, B.edge_index, B.batch
        for conv in self.convs:
            x = conv(x, edge_index).relu()
        x = global_add_pool(x, batch)
        return F.tanh(self.mlp(x))


class MLP(nn.Module):
    def __init__(self, num_layers, layer_size, input_size, output_size, dropout_p, use_batchnorm):
        super(MLP, self).__init__()

        self.layers = nn.ModuleList()
        self.use_batchnorm = use_batchnorm

        # add input layer
        self.layers.append(nn.Linear(input_size, layer_size))
        if self.use_batchnorm:
            self.layers.append(nn.BatchNorm1d(layer_size))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_p))

        # add hidden layers
        for i in range(num_layers - 2):
            self.layers.append(nn.Linear(layer_size, layer_size))
            if self.use_batchnorm:
                self.layers.append(nn.BatchNorm1d(layer_size))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_p))

        # add output layer
        self.layers.append(nn.Linear(layer_size, output_size))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return torch.tanh(x)
