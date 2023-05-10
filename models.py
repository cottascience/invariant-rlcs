import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import MLP as torch_geometric_MLP

def dot( x, a ):
    return torch.bmm(x.view(x.shape[0], 1, x.shape[1]), a.view(a.shape[0], a.shape[1], 1)).reshape(x.shape[0])

class RLC(torch.nn.Module):
     def __init__(self, noise_size, hidden_size, num_layers, dropout_p, use_batchnorm, x_size):
         super(RLC, self).__init__()

         norm = 'batch_norm' if use_batchnorm else None
         self.mlp = torch_geometric_MLP(in_channels = noise_size, hidden_channels = hidden_size, out_channels = x_size+1,
                        num_layers=num_layers, norm=norm, dropout=dropout_p)
         self.noise_size = noise_size
     def forward(self, x):
         noise = torch.rand(( x.shape[0], self.noise_size )).to(x.device)
         out = self.mlp( noise  )
         a = out[:,:-1]
         b = out[:,-1].unsqueeze(1)
         print(b.shape)
         print(a.shape)
         print(x.shape)
         print( dot(x,a).shape  )
         exit()
         return dot(x,a) - b

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
        return self.mlp(x)


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
        return x
