from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch
import numpy as np
from torch_geometric.utils import negative_sampling


class DynGraphNode:
    def __init__(self, snapshot_num, edge, node, Xt, Yt, edgeT, nodeT):
        self.snapshot_num = snapshot_num
        self.edge = edge
        self.node = node
        self.Xt = Xt
        self.Yt = Yt
        self.nodeT = np.concatenate((
            np.array([0]), np.cumsum(nodeT)))
        self.edgeT = np.concatenate((
            np.array([0]), np.cumsum(edgeT)))
        self.edgeT -= (self.edgeT % 2 == 1)  # 保证每个snapshot都是无向图

    def getFullEdge(self, t: int):
        return self.edge[:, :self.edgeT[t+1]]

    def getDiffEdge(self, t: int):
        return self.edge[:, self.edgeT[t]:self.edgeT[t+1]]

    def getFullNode(self, t: int):
        return self.node[:self.nodeT[t+1]]

    def getDiffNode(self, t: int):
        return self.node[self.nodeT[t]:self.nodeT[t+1]]

    def getTrainSnapshot(self, t: int, mode="GCN"):
        if mode == "GCN":
            return self.Xt[t], self.Yt[t][self.getFullNode(t)], self.getFullEdge(t), self.getFullNode(t)
        else:
            return self.Xt[t][self.getFullNode(t)], self.Yt[t][self.getFullNode(t)]

    def getTestSnapshot(self, t: int, mode="GCN"):
        if mode == "GCN":
            return self.Xt[t], self.Yt[t][self.getDiffNode(t)], self.getFullEdge(t), self.getFullNode(t)
        else:
            return self.Xt[t][self.getDiffNode(t)], self.Yt[t][self.getDiffNode(t)]
# only int edge value


class DynGraphEdge:
    def __init__(self, snapshot_num, edge, node, Xt, edgevalT, edgeT, nodeT):
        self.snapshot_num = snapshot_num
        self.edge = edge  # format: node1,node2,val
        self.node = node
        self.Xt = Xt
        self.nodeT = np.concatenate((
            np.array([0]), np.cumsum(nodeT)))
        self.edgeT = np.concatenate((
            np.array([0]), np.cumsum(edgeT)))
        self.edgeT -= (self.edgeT % 2 == 1)  # 保证每个snapshot都是无向图
        self.edgevalT = edgevalT

    def getFullEdge(self, t: int):
        return self.edge[:, :self.edgeT[t+1]], self.edgevalT[t][:self.edgeT[t+1]]

    def getDiffEdge(self, t: int):
        return self.edge[:, self.edgeT[t]:self.edgeT[t+1]], self.edgevalT[t][self.edgeT[t]:self.edgeT[t+1]]

    def getFullNode(self, t: int):
        return self.node[:self.nodeT[t+1]]

    def getDiffNode(self, t: int):
        return self.node[self.nodeT[t]:self.nodeT[t+1]]

    def getTrainSnapshot(self, t: int, mode="LP"):
        if mode == "LP":
            posEdge, _ = self.getFullEdge(t)[:2]
            negEdge = negative_sampling(
                posEdge, num_nodes=self.node.shape[0]).numpy()
            Y = np.concatenate(
                (np.ones((posEdge.shape[1]), dtype=np.int),
                 np.zeros((negEdge.shape[1]), dtype=np.int)))
            edge1 = np.concatenate((posEdge[0], negEdge[0]))
            edge2 = np.concatenate((posEdge[1], negEdge[1]))
            return self.Xt[t][edge1], self.Xt[t][edge2], Y
        elif mode == "EC":
            edge, Y = self.getFullEdge(t)
            return self.Xt[t][edge[0]], self.Xt[t][edge[1]], Y
        elif mode == "GCNLP":
            posEdge, _ = self.getFullEdge(t)[:2]
            negEdge = negative_sampling(
                posEdge, num_nodes=self.node.shape[0]).numpy()
            Y = np.concatenate(
                (np.ones((posEdge.shape[1]), dtype=np.int),
                 np.zeros((negEdge.shape[1]), dtype=np.int)))
            edge1 = np.concatenate((posEdge[0], negEdge[0]))
            edge2 = np.concatenate((posEdge[1], negEdge[1]))
            return self.Xt[t], posEdge, edge1, edge2, Y
        elif mode == "GCNEC":
            edge, Y = self.getFullEdge(t)
            return self.Xt[t], edge, edge[0], edge[1], Y

    def getTestSnapshot(self, t: int, mode="LP"):
        if mode == "LP":
            edge, _ = self.getFullEdge(t)[:2]
            posEdge, _ = self.getDiffEdge(t)[:2]
            negEdge = negative_sampling(
                edge, num_neg_samples=posEdge.shape[1],
                num_nodes=self.node.shape[0]).numpy()
            Y = np.concatenate(
                (np.ones((posEdge.shape[1]), dtype=np.int),
                 np.zeros((negEdge.shape[1]), dtype=np.int)))
            edge1 = np.concatenate((posEdge[0], negEdge[0]))
            edge2 = np.concatenate((posEdge[1], negEdge[1]))
            return self.Xt[t][edge1], self.Xt[t][edge2], Y
        elif mode == "EC":
            edge, Y = self.getDiffEdge(t)
            return self.Xt[t][edge[0]], self.Xt[t][edge[1]], Y
        elif mode == "GCNLP":
            edge, _ = self.getFullEdge(t)[:2]
            posEdge, _ = self.getDiffEdge(t)[:2]
            negEdge = negative_sampling(
                edge, num_neg_samples=posEdge.shape[1],
                num_nodes=self.node.shape[0]).numpy()
            Y = np.concatenate(
                (np.ones((posEdge.shape[1]), dtype=np.int),
                 np.zeros((negEdge.shape[1]), dtype=np.int)))
            edge1 = np.concatenate((posEdge[0], negEdge[0]))
            edge2 = np.concatenate((posEdge[1], negEdge[1]))
            return self.Xt[t], edge, edge1, edge2, Y
        elif mode == "GCNEC":
            edge, _ = self.getFullEdge(t)
            sample_edge, Y = self.getDiffEdge(t)
            return self.Xt[t], edge, sample_edge[0], sample_edge[1], Y


def getEdgeWeight(edge):
    return torch.ones((edge.shape[1]), dtype=torch.float)


def divideArray(arr, ratio: float):
    a1, a2 = train_test_split(arr, train_size=ratio, test_size=1-ratio)
    return a1


def splitIndex(arr, ratio: float):
    a1, a2 = train_test_split(arr, train_size=ratio, test_size=1-ratio)
    return a1, a2


def divideTensorDataset(dataset, rTrain: float, rValid: float):
    return Data.random_split(dataset,
                             [int(rTrain*len(dataset)),
                              int(rValid*len(dataset)),
                                 len(dataset)-int(rTrain*len(dataset))-int(rValid*len(dataset))])


def stratified_divideXY(X, Y, rTrain: float, rValid: float):
    X_train, X_t, Y_train, Y_t = train_test_split(
        X, Y, train_size=rTrain, stratify=Y)
    X_valid, X_test, Y_valid, Y_test = train_test_split(
        X_t, Y_t, train_size=rValid/(1-rTrain), stratify=Y_t)
    return Data.TensorDataset(
        torch.Tensor(X_train).to(torch.float),
        torch.Tensor(Y_train).to(torch.long)
    ), Data.TensorDataset(
        torch.Tensor(X_valid).to(torch.float),
        torch.Tensor(Y_valid).to(torch.long)
    ), Data.TensorDataset(
        torch.Tensor(X_test).to(torch.float),
        torch.Tensor(Y_test).to(torch.long))


def diedge2unedge(dedge):
    return np.concatenate((dedge[[0, 1]], dedge[[1, 0]]), axis=0).transpose().reshape(-1, 2).transpose()
