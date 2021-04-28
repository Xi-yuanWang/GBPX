from sklearn.model_selection import train_test_split
import torch.utils.data as Data
import torch


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
