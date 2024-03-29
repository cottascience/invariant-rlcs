import argparse
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
import models
import sys
import datasets
import torch_geometric
from torch.utils.data.dataloader import default_collate

# Parse the hyperparameters
parser = argparse.ArgumentParser(description='experiments for randomized linear classifiers')
parser.add_argument('--input_size', type=int, default=8, help='dimension of input')
parser.add_argument('--hidden_size', type=int, default=8, help='number of hidden units in each layer in the MLP')
parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers in the MLP')
parser.add_argument('--train_size', type=int, default=250, help='number of training examples')
parser.add_argument('--test_size', type=int, default=100, help='number of testing examples')
parser.add_argument('--lr', type=float, default=5e-1, help='initial learning rate for the optimizer')
parser.add_argument('--epochs', type=int, default=1000, help='number of epochs to train for')
parser.add_argument('--runs', type=int, default=5, help='number of runs')
parser.add_argument('--patience', type=int, default=15, help='patience')
parser.add_argument('--batch_size', type=int, default=250, help='mini-batch size for training')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout probability for the MLP')
parser.add_argument('--weight_decay', type=float, default=0.00, help='weight decay for L2 regularization')
parser.add_argument('--m', type=int, default=100, help='number of samples used in RLCs for eval')
parser.add_argument('--k', type=int, default=100, help='number of samples used in RLCs for train')
parser.add_argument('--noise_size', type=int, default=1, help='number of noise variables in RLCs')
parser.add_argument('--model', choices=['mlp', 'gnn', 'deepsets' ,'rlc', 'rlc_set', 'rlc_graph', 'rlc_sphere'], default='mlp')
parser.add_argument('--dataset', choices=['ball', 'parity', 'sort', 'range' , 'connectivity', 'sin'], default='ball')
parser.add_argument('--OOD', action='store_true')

args = parser.parse_args()

print('---Settings being used---')
print(args)
print('-------------------------')
train_results = []
test_results = []

for run in range(args.runs):
    # Init the model
    if args.model == 'mlp': model = models.MLP( num_layers=args.num_layers, layer_size = args.hidden_size, input_size=args.input_size, output_size=1, dropout_p=args.dropout, use_batchnorm=True )
    if args.model == 'gnn': model = models.GIN( in_channels=1, hidden_channels=args.hidden_size, out_channels=1, num_layers=args.num_layers  )
    if args.model == 'rlc': model = models.RLC( noise_size=args.noise_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_p=args.dropout, use_batchnorm=True, x_size=args.input_size )
    if args.model == 'rlc_sphere': model = models.RSphereC( noise_size=args.noise_size, hidden_size=args.hidden_size, num_layers=args.num_layers, dropout_p=args.dropout, use_batchnorm=True, x_size=args.input_size )
    if args.model == 'deepsets': model = models.DeepSets(hidden_size=args.hidden_size, num_layers=args.num_layers,
                                                         dropout_p=args.dropout, use_batchnorm=True)
    if args.model == 'rlc_set': model = models.RSetC( noise_size=args.noise_size, hidden_size=args.hidden_size, num_layers=args.num_layers,
                                               dropout_p=args.dropout, use_batchnorm=True, x_size=args.input_size )
    if args.model == 'rlc_graph': model = models.RGraphC( hidden_size=args.hidden_size, num_layers=args.num_layers,
                                                dropout_p=args.dropout, use_batchnorm=True, x_size=args.input_size )

    if torch.cuda.is_available(): model = model.cuda()

    # Define the loss function (hinge loss)
    criterion = nn.SoftMarginLoss()

    # Define the optimizer with L2 regularization
    optimizer = optim.Adagrad(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = args.epochs)
    # Create the data loaders
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)
    datasets_dict = { 'ball': datasets.Ball, 'parity': datasets.Parity , 'range': datasets.Range ,'sort': datasets.Sort , 'connectivity': datasets.Connectivity, 'sin': datasets.Sin }
    print('Creating training data')
    train_dataset = datasets_dict[args.dataset]( args.train_size, args.input_size  )
    print('Creating validation data')
    val_dataset = datasets_dict[args.dataset]( args.test_size, args.input_size  )
    print('Creating test data')
    test_dataset = datasets_dict[args.dataset]( args.test_size, args.input_size, log=True  )
    train_loader = DataLoader(dataset = train_dataset, batch_size = args.batch_size, shuffle=True)
    val_loader = DataLoader(dataset = val_dataset, batch_size = args.batch_size)
    test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size)
    if args.OOD:
        test_dataset = datasets_dict[args.dataset]( args.test_size, args.input_size*2 + 1 , log=True)
        test_loader = DataLoader(dataset = test_dataset, batch_size = args.batch_size)
    if args.model == 'gnn':
        train_dataset.use_graphs = True
        val_dataset.use_graphs = True
        test_dataset.use_graphs = True
    seed = random.randint(0, sys.maxsize)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    def graph_batch(x):
        G = [ torch_geometric.utils.dense_to_sparse(g) for g in x.to_dense() ]
        if args.OOD:
            x = torch_geometric.data.Batch().from_data_list( [ torch_geometric.data.Data( edge_index=g[0], x = torch.ones((2*args.input_size+1,1)))  for g in G  ]  )
        else:
            x = torch_geometric.data.Batch().from_data_list( [ torch_geometric.data.Data( edge_index=g[0], x = torch.ones((args.input_size,1)))  for g in G  ]  )
        return x

    def accuracy( model, loader ):
        model.eval()
        correct, total = 0, 0
        for x,y in loader:
            total += x.shape[0]
            if len(x.shape) == 3:
                y = y.float()
                x = graph_batch(x)
            if torch.cuda.is_available(): x,y = x.cuda(), y.cuda()
            y_hat = torch.zeros_like(y)
            for _ in range(args.m):
                y_hat += torch.sign(model(x))
            y_hat = torch.sign(y_hat)
            correct += torch.sum(y_hat == y)
        model.train()
        return correct/total

    # Train the model
    best_val, best_train, best_test = 0, 0, 0
    patience = args.patience
    for epoch in range(args.epochs):
        epoch_loss, epoch_size  = 0, 0
        for x,y in train_loader:
            num_ex = x.shape[0]
            if len(x.shape) == 3:
                x = graph_batch(x)
            else:
                x,y = x.repeat(args.k,1), y.repeat(args.k,1)
            if torch.cuda.is_available(): x,y = x.cuda(), y.cuda()
            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            epoch_size += num_ex
            epoch_loss += loss.item()*num_ex
        scheduler.step()
        val_acc = accuracy(model, val_loader).item()
        patience -= 1
        if val_acc > best_val:
            best_val = val_acc
            best_train = accuracy(model, train_loader).item()
            best_test = accuracy(model, test_loader).item()
            patience = args.patience
        if patience == 0: break
        print(epoch, '==\t Loss:\t', epoch_loss/epoch_size, 'LR:\t',scheduler.get_lr(),'Train acc:\t', accuracy(model, train_loader).item(), 'Val acc:\t', val_acc, 'Test acc:\t', accuracy(model, test_loader).item(), 'Patience:\t', patience, best_val)

    # Save the performance
    train_results.append( best_train  )
    test_results.append( best_test )

print( "Final train results:", np.mean(np.array( train_results ) ) , np.std(np.array( train_results ) ) )
print( "Final test results:", np.mean(np.array( test_results ) ) , np.std(np.array( test_results ) ))

