import torch
import torch.nn as nn
import torch.nn.functional as F
from .model import accuracy, multiclass_f1


class LSTMConvMLP(nn.Module):
    def __init__(self, depth, nF, num_layers, hidden_channels,
                 out_channels, dropout):
        super(LSTMConvMLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.conv_layer = torch.nn.Conv1d(nF, nF, depth, groups=nF)
        self.recurrent_layer = torch.nn.LSTM(depth*nF, depth*nF, num_layers=1)
        self.lins.append(torch.nn.Linear(nF, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        self.conv_layer.reset_parameters()
        self.recurrent_layer.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        w = self.conv_layer.weight
        wshape = w.shape
        w, _ = self.recurrent_layer(w.reshape(1, 1, -1))
        self.conv_layer.weight = torch.nn.Parameter(
            w.reshape(wshape[0], wshape[1], wshape[2]))
        x = self.conv_layer(x).reshape(x.shape[0], -1)
        x = F.relu(x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


def trainMLP(mod, snapshot_list, dynGraph, optimizer, loss_fn=nn.NLLLoss()):
    mod.train()
    cost = 0
    for t in snapshot_list:
        X, Y = dynGraph.getTrainSnapshot(t, mode="MLP")
        X = torch.from_numpy(X).to(torch.float)
        Y = torch.from_numpy(Y).to(torch.long)
        y_hat = mod(X)
        cost += loss_fn(y_hat, Y)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    return cost.item()


@torch.no_grad()
def testMLP(mod, snapshot_list, dynGraph, score_fn=multiclass_f1):
    mod.eval()
    for t in snapshot_list:
        X, Y = dynGraph.getTestSnapshot(t, mode="MLP")
        X = torch.from_numpy(X).to(torch.float)
        Y = torch.from_numpy(Y).to(torch.long)
        y_hat = mod(X)
        print("T=", t, "score=", score_fn(y_hat, Y))


def testX(X, Y, dynGraph, trainSnapshot, validSnapshot, testSnapshot,
          niter=20, classifer=None, loss_fn=torch.nn.NLLLoss(),
          batch_size=1024, score_fn=multiclass_f1):
    if(classifer is None):
        classifer = LSTMConvMLP(
            X.shape[2], X.shape[1], 2, 128, len(set(Y)), 0.5)

    optimizer = torch.optim.Adam(classifer.parameters(), lr=0.01)

    for i in range(niter):
        trainloss = trainMLP(classifer, trainSnapshot,
                             dynGraph, optimizer, loss_fn)
        print("trainloss=", trainloss)
        testMLP(classifer, validSnapshot, dynGraph, score_fn)
    testMLP(classifer, testSnapshot, dynGraph, score_fn)
    return classifer
