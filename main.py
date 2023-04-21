import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import functorch

# Set the seed for reproducibility
torch.manual_seed(42)

# Parse the hyperparameters
parser = argparse.ArgumentParser(description='experiments for randomized linear classifiers')
parser.add_argument('--input-size', type=int, default=8, help='dimension of input')
parser.add_argument('--hidden-size', type=int, default=16, help='number of hidden units in the MLP')
parser.add_argument('--train-size', type=int, default=10000, help='number of training examples')
parser.add_argument('--test-size', type=int, default=1000, help='number of testing examples')
parser.add_argument('--lr', type=float, default=0.1, help='initial learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--batch-size', type=int, default=64, help='mini-batch size for training')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability for the MLP')
parser.add_argument('--weight-decay', type=float, default=0.01, help='weight decay for L1 regularization')
args = parser.parse_args()

# Generate random bitstrings for the training set
train_input = torch.randint(low=0, high=2, size=(args.train_size, args.input_size)).float()
train_target = train_input.sum(dim=1) % 2  # Compute the parity of each bitstring

# Generate random bitstrings for the testing set
test_input = torch.randint(low=0, high=2, size=(args.test_size, args.input_size)).float()
test_target = test_input.sum(dim=1) % 2  # Compute the parity of each bitstring

# Define the MLP model
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(args.input_size, args.hidden_size)
        self.bn1 = nn.BatchNorm1d(args.hidden_size)
        self.dropout1 = nn.Dropout(p=args.dropout)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.bn2 = nn.BatchNorm1d(args.hidden_size)
        self.dropout2 = nn.Dropout(p=args.dropout)
        self.fc3 = nn.Linear(args.hidden_size, args.output_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# Initialize the MLP model
model = MLP()

# Define the loss function (hinge loss)
criterion = nn.HingeEmbeddingLoss()

# Define the optimizer (Stochastic Gradient Descent) with L1 regularization
optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

# Define the learning rate scheduler (Cosine Annealing)
scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

# Train the MLP model
for epoch in range(args.epochs):
    epoch_loss = 0.
    epoch_accuracy = 0.

    # Shuffle the training set
    permutation = torch.randperm(args

