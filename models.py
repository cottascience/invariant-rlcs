import torch
import torch.nn as nn
from torch_geometric.nn import GINConv, global_add_pool
from torch_geometric.nn import MLP as torch_geometric_MLP

class GIN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers):
        super(GIN).__init__()

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            mlp = torch_geometric_MLP([in_channels, hidden_channels, hidden_channels])
            self.convs.append(GINConv(nn=mlp, train_eps=False))
            in_channels = hidden_channels

        self.mlp = torch_geometric_MLP([hidden_channels, hidden_channels, out_channels],
                       norm=None, dropout=0.5)

    def forward(self, x, edge_index, batch):
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
