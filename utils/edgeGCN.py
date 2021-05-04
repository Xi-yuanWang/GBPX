from torch_geometric_temporal.nn.recurrent import EvolveGCNO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model import multiclass_f1, MLP
from .util import getEdgeWeight


class EGCNO(torch.nn.Module):
    def __init__(self, inChannel):
        super(EGCNO, self).__init__()
        self.recurrent_1 = EvolveGCNO(inChannel)
        self.recurrent_2 = EvolveGCNO(inChannel)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        return x


class GCNEdgeClassifier(nn.Module):
    def __init__(self, inChannel, hidden_channels, num_layer, num_class, dropout):
        super(GCNEdgeClassifier, self).__init__()
        self.GCN = EGCNO(inChannel)
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(inChannel, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layer - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, num_class))
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, X, edge_index, edge_weight, edge1, edge2):
        X = self.GCN(X, edge_index, edge_weight)
        X1 = X[edge1]
        X2 = X[edge2]
        x = X1*X2
        for i, lin in enumerate(self.lins[0:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)



def trainGCN(mod, snapshot_list, dynGraph, optimizer, mode="GCNLP", loss_fn=nn.NLLLoss()):
    mod.train()
    cost = 0
    for t in snapshot_list:
        X, edge_index, edge1, edge2, Y = dynGraph.getTrainSnapshot(t, mode)
        X = torch.from_numpy(X).to(torch.float)
        Y = torch.from_numpy(Y).to(torch.long)
        edge_index = torch.from_numpy(edge_index).to(torch.long)
        edge1 = torch.from_numpy(edge1).to(torch.long)
        edge2 = torch.from_numpy(edge2).to(torch.long)
        y_hat = mod(X, edge_index, getEdgeWeight(edge_index), edge1, edge2)
        cost += loss_fn(y_hat, Y)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    return cost.item()


@torch.no_grad()
def testGCN(mod, snapshot_list, dynGraph, mode="GCNLP", score_fn=multiclass_f1):
    mod.eval()
    for t in snapshot_list:
        X, edge_index, edge1, edge2, Y = dynGraph.getTestSnapshot(t, mode)
        X = torch.from_numpy(X).to(torch.float)
        Y = torch.from_numpy(Y).to(torch.long)
        edge_index = torch.from_numpy(edge_index).to(torch.long)
        edge1 = torch.from_numpy(edge1).to(torch.long)
        edge2 = torch.from_numpy(edge2).to(torch.long)
        y_hat = mod(X, edge_index, getEdgeWeight(edge_index), edge1, edge2)
        print("T=", t, "score=", score_fn(y_hat, Y))
