from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data as Data


class MLP(nn.Module):
    def __init__(self, in_channels, num_layers, hidden_channels, out_channels,
                 dropout):
        super(MLP, self).__init__()

        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


class ConvMLP(nn.Module):
    def __init__(self, depth, nF, num_layers, hidden_channels,
                 out_channels, dropout):
        super(ConvMLP, self).__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Conv1d(nF, nF, depth, groups=nF))
        self.lins.append(torch.nn.Linear(nF, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        x = self.lins[0](x).reshape(x.shape[0], -1)
        x = F.relu(x)
        for i, lin in enumerate(self.lins[1:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)


class Dense(nn.Module):
    def __init__(self, in_features, out_features, bias='bn'):
        super(Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        if bias == 'bn':
            self.bias = nn.BatchNorm1d(out_features)
        else:
            self.bias = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input):
        output = torch.mm(input, self.weight)
        output = self.bias(output)
        if self.in_features == self.out_features:  # ??
            output = output + input
        return output


class GnnBP(nn.Module):
    def __init__(self, nfeat, nlayers,
                 nhidden, nclass, dropout, mode='multi_class', bias='bn'):
        super(GnnBP, self).__init__()
        self.fcs = nn.ModuleList()
        self.fcs.append(Dense(nfeat, nhidden, bias))
        for _ in range(nlayers-2):
            self.fcs.append(Dense(nhidden, nhidden, bias))
        self.fcs.append(Dense(nhidden, nclass))
        self.act_fn = nn.ReLU()
        self.dropout = dropout
        if mode == 'multi_class':
            self.out_fn = nn.LogSoftmax(dim=-1)
        else:
            self.out_fn = nn.Sigmoid()

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.act_fn(self.fcs[0](x))
        for fc in self.fcs[1:-1]:
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.act_fn(fc(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.fcs[-1](x)
        x = self.out_fn(x)
        return x


def multilabel_f1(y_true, y_pred):
    y_pred[y_pred > 0.5] = 1
    y_pred[y_pred <= 0.5] = 0
    y_true = (y_true > 0.5)
    return f1_score(y_true, y_pred, average="micro")


def accuracy(output, labels):
    output = np.argmax(output, axis=1)
    micro = accuracy_score(labels, output)
    return micro


def multiclass_f1(output, labels):
    output = np.argmax(output, axis=1)
    micro = f1_score(labels, output, average='micro')
    return micro


def train(model, loader, loss_fn, optimizer, batch_cnt=-1):
    model.train()
    loss_list = []
    for step, (batch_x, batch_y) in enumerate(loader):
        if(step == batch_cnt):
            return loss_list
        optimizer.zero_grad()
        output = model(batch_x)
        loss_train = loss_fn(output, batch_y)
        loss_train.backward()
        optimizer.step()
        loss_list.append(loss_train.item())
    return loss_list


@torch.no_grad()
def test(model, loader, loss_fn, score_fn, batch_cnt=-1):
    model.eval()
    score_list = []
    loss_list = []
    for step, (batch_x, batch_y) in enumerate(loader):
        if(step == batch_cnt):
            return score_list
        output = model(batch_x)
        loss_list.append(loss_fn(output, batch_y).item())
        score_list.append(score_fn(output.detach().numpy(),
                                   batch_y.detach().numpy()))
    return loss_list, score_list

# 一个简单的测试预计算结果的函数。split是指结点的下标组成的array


def testX_split(X, Y, train_split, valid_split, test_split,
                niter=20, classifer=None, loss_fn=torch.nn.NLLLoss(),
                batch_size=1024, score_fn=accuracy, optimizer=None):
    if(classifer is None):
        classifer = ConvMLP(X.shape[2], X.shape[1], 2, 128, len(set(Y)), 0.5)
    trainset = Data.TensorDataset(torch.Tensor(X[train_split]).to(
        torch.float), torch.Tensor(Y[train_split]).to(torch.long))
    validset = Data.TensorDataset(torch.Tensor(X[valid_split]).to(
        torch.float), torch.Tensor(Y[valid_split]).to(torch.long))
    testset = Data.TensorDataset(torch.Tensor(X[test_split]).to(
        torch.float), torch.Tensor(Y[test_split]).to(torch.long))
    trainloader = Data.DataLoader(dataset=trainset,
                                  batch_size=batch_size,
                                  shuffle=True)
    validloader = Data.DataLoader(dataset=validset,
                                  batch_size=8192,
                                  shuffle=True)
    testloader = Data.DataLoader(dataset=testset,
                                 batch_size=8192,
                                 shuffle=True)

    score = 0
    if(optimizer is None):
        optimizer = torch.optim.Adam(classifer.parameters(), lr=0.01)
    for i in range(niter):
        trainloss = train(
            classifer, trainloader, loss_fn, optimizer)
        validloss, validscore = test(
            classifer, validloader, loss_fn, score_fn)
        score = np.average(validscore)
        print(score)
    testloss, testscore = test(
        classifer, testloader, loss_fn, score_fn)
    print(np.average(testscore))
    return classifer
