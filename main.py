import argparse
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import models
import datasets

# Parse the hyperparameters
parser = argparse.ArgumentParser(description='experiments for randomized linear classifiers')
parser.add_argument('--input_size', type=int, default=8, help='dimension of input')
parser.add_argument('--hidden_size', type=int, default=16, help='number of hidden units in each layer in the MLP')
parser.add_argument('--num_layers', type=int, default=2, help='number of hidden layers in the MLP')
parser.add_argument('--train_size', type=int, default=10000, help='number of training examples')
parser.add_argument('--test_size', type=int, default=1000, help='number of testing examples')
parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size for training')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability for the MLP')
parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay for L2 regularization')
parser.add_argument('--margin', type=float, default=1.00, help='margin in hinge loss')
args = parser.parse_args()

# Init the model
model = models.MLP( num_layers=args.num_layers, layer_size = args.hidden_size, input_size=args.input_size, output_size=1, dropout_p=args.dropout, use_batchnorm=True )

# Define the loss function (hinge loss)
criterion = nn.HingeEmbeddingLoss(margin=args.margin)

# Define the optimizer (Stochastic Gradient Descent) with L2 regularization
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Create the data loaders
train_dataset = datasets.Ball( args.train_size, args.input_size  )
test_dataset = datasets.Ball( args.test_size, args.input_size  )
train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size)
test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size)

def accuracy( model, loader ):
    model.eval()
    correct = 0
    total = 0
    for x,y in loader:
        y_hat = torch.sign(model(x))
        correct += int((y_hat == y).sum())
        total += x.shape[0]
    model.train()
    return correct/total

# Train the MLP model
for epoch in range(args.epochs):
    for x,y in train_loader:
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
    print(accuracy(model, train_loader), accuracy(model, test_loader) , loss)
