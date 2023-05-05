import torch
import torch.nn as nn

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
