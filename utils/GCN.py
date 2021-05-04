from torch_geometric_temporal.nn.recurrent import EvolveGCNO
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .model import multiclass_f1
from .util import getEdgeWeight


class EGCNO(torch.nn.Module):
    def __init__(self, inChannel, num_classes):
        super(EGCNO, self).__init__()
        self.recurrent_1 = EvolveGCNO(inChannel)
        self.recurrent_2 = EvolveGCNO(inChannel)
        self.linear = torch.nn.Linear(inChannel, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = self.recurrent_1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.recurrent_2(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.linear(x)
        return F.log_softmax(x, dim=-1)


def trainGCN(mod, snapshot_list, dynGraph, optimizer, loss_fn=nn.NLLLoss()):
    mod.train()
    cost = 0
    for t in snapshot_list:
        X, Y, edge_index, node = dynGraph.getSnapshot(t)
        X = torch.from_numpy(X).to(torch.float)
        Y = torch.from_numpy(Y).to(torch.long)
        edge_index = torch.from_numpy(edge_index).to(torch.long)
        node = torch.from_numpy(node).to(torch.long)
        y_hat = mod(X, edge_index, getEdgeWeight(edge_index))
        cost += loss_fn(y_hat[node], Y)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    return cost.item()


@torch.no_grad()
def testGCN(mod, snapshot_list, dynGraph, score_fn=multiclass_f1):
    mod.eval()
    for t in snapshot_list:
        X, Y, edge_index, node = dynGraph.getSnapshot(t)
        X = torch.from_numpy(X).to(torch.float)
        Y = torch.from_numpy(Y).to(torch.long)
        edge_index = torch.from_numpy(edge_index).to(torch.long)
        node = torch.from_numpy(node).to(torch.long)
        y_hat = mod(X, edge_index, getEdgeWeight(edge_index))
        y_hat = y_hat[node]
        print("T=", t, "score=", score_fn(y_hat, Y))
