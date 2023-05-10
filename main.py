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
parser.add_argument('--input_size', type=int, default=16, help='dimension of input')
parser.add_argument('--hidden_size', type=int, default=16, help='number of hidden units in each layer in the MLP')
parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers in the MLP')
parser.add_argument('--train_size', type=int, default=1000, help='number of training examples')
parser.add_argument('--test_size', type=int, default=1000, help='number of testing examples')
parser.add_argument('--lr', type=float, default=1e-2, help='initial learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--batch_size', type=int, default=64, help='mini-batch size for training')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability for the MLP')
parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay for L2 regularization')
parser.add_argument('--m', type=int, default=100, help='number of samples used in RLCs for eval')
parser.add_argument('--k', type=int, default=100, help='number of samples used in RLCs for train')
parser.add_argument('--noise_size', type=int, default=16, help='number of noise variables in RLCs')
parser.add_argument('--model', choices=['mlp', 'gnn', 'rlc', 'rlc_set', 'rlc_graph', 'rlc_sphere'], default='mlp')
parser.add_argument('--dataset', choices=['ball', 'parity', 'sort', 'connectivity'], default='ball')
args = parser.parse_args()
# python main.py --dataset ball --model mlp
# python main.py --dataset ball --model rlc

print('---Settings being used---')
print(args)
print('-------------------------')
# Init the model
if args.model == 'mlp': model = models.MLP( num_layers=args.num_layers, layer_size = args.hidden_size, input_size=args.input_size, output_size=1, dropout_p=args.dropout, use_batchnorm=True )
if args.model == 'gnn': models.GIN( in_channels=args.input_size, hidden_channels=args.hidden_size, out_channels=1, num_layers=args.num_layers  )
if args.model == 'rlc': model = models.RLC( noise_size=args.noise_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_p=args.dropout, use_batchnorm=True, x_size=args.input_size )

if torch.cuda.is_available(): model = model.cuda()

# Define the loss function (hinge loss)
# criterion = nn.HingeEmbeddingLoss(margin=args.margin)
criterion = nn.SoftMarginLoss(reduction='none')

# Define the optimizer (Stochastic Gradient Descent) with L2 regularization
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Create the data loaders
datasets = { 'ball': datasets.Ball, 'parity': datasets.Parity , 'sort': datasets.Sort , 'connectivity': datasets.Connectivity  }
print('Creating training data')
train_dataset = datasets[args.dataset]( args.train_size, args.input_size  )
print('Creating test data')
test_dataset = datasets[args.dataset]( args.test_size, args.input_size  )
train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size)
test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size)
if args.model == 'gnn':
    train_dataset.use_graphs
    test_dataset.use_graphs

def accuracy( model, loader ):
    model.eval()
    correct = 0
    for x,y in loader:
        if torch.cuda.is_available(): x,y = x.cuda(), y.cuda()
        y_hat = torch.zeros_like(y)
        for _ in range(args.m):
            y_hat += torch.sign(model(x))
        y_hat = torch.sign(y_hat)
        correct += torch.sum(y_hat == y)
    model.train()
    return correct/args.train_size

# Train the MLP model
for epoch in range(args.epochs):
    epoch_loss, epoch_size  = 0, 0
    for x,y in train_loader:
        if torch.cuda.is_available(): x,y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        weights = torch.zeros_like(y)
        for _ in range(args.k):
            weights += torch.sign(model(x))
        weights /= args.k
        loss = torch.mean(-weights*loss)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*x.shape[0]
        epoch_size += x.shape[0]
    print(epoch, '==\t Loss:\t', epoch_loss/epoch_size, 'Train acc:\t', accuracy(model, train_loader).item(), 'Test acc:\t', accuracy(model, test_loader).item())
