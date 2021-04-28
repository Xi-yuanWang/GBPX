import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from utils.util import divideArray
# Dense Feature
# directed??


class Graph:
    nV: int
    nF: int
    r: float
    wl: np.ndarray
    L: int
    X: np.ndarray
    P: np.ndarray

    def __init__(self, nV: int, nF: int, r: float, wl:
                 np.ndarray, L: int):
        self.nV = nV
        self.nF = nF
        self.r = r
        self.wl = wl
        self.L = L

    def setEdge(self, edge_index: np.ndarray):
        self.A = sp.sparse.csr_matrix(
            (np.ones((edge_index.shape[1])),
             (edge_index[0],
              edge_index[1])), shape=(self.nV, self.nV))
        self.D = np.array(self.A.sum(axis=1)).flatten()
        self.D += self.D<0.5# 保证不出现inf
        self.Trev = sp.sparse.diags(1/self.D)*self.A

    def setDenseX(self, X: np.ndarray):
        self.rawX = X
        self.X = X

    def setSparseX(self, X: np.ndarray):
        self.rawX = sp.sparse.csr_matrix(X)
        self.X = X

    def reset(self):
        self.X = self.rawX
        self.nF = self.X.shape[1]

    def compressSparse(self, dim: int, pca_train_ratio: float):
        self.nF = dim
        self.pca = TruncatedSVD(n_components=dim, n_iter=10)
        self.pca.fit(divideArray(self.rawX, pca_train_ratio))
        self.X = self.pca.transform(self.rawX)
        return self.pca.explained_variance_ratio_

    def compressDense(self, dim: int, pca_train_ratio: float):
        self.nF = dim
        self.pca = PCA(n_components=dim, iterated_power=10)
        self.pca.fit(divideArray(self.rawX, pca_train_ratio))
        self.X = self.pca.transform(self.rawX)
        return self.pca.explained_variance_ratio_

    def precompute(self):
        self.Dnegr = np.power(self.D, -self.r).reshape(-1, 1)
        self.Q = np.zeros((self.L+1, self.nV, self.nF))
        np.multiply(self.Dnegr, self.X, out=self.Q[0])
        for i in range(self.L):
            self.Q[i+1] = self.Trev.dot(self.Q[i])
        self.getP(self.wl)

    def concatenateQ(self):
        return (1/self.Dnegr)*np.swapaxes(self.Q, 0, 1).reshape(self.nV, -1)

    def concatenateQForConv(self):
        return (1/self.Dnegr).reshape(-1, 1, 1)*np.transpose(self.Q, (1, 2, 0))

    # reset wl and get P
    def getP(self, wl):
        self.P = (1/self.Dnegr) * \
            np.tensordot(self.wl[::-1], self.Q, axes=[0, 0])


class DiGraph:
    nV: int
    nF: int
    r: float
    wl: np.ndarray
    L: int
    X: np.ndarray
    P: np.ndarray

    def __init__(self, nV: int, nF: int, r: float, wl:
                 np.ndarray, L: int):
        self.nV = nV
        self.nF = nF
        self.r = r
        self.wl = wl
        self.L = L
    # 有向图中的边正向输入

    def setForwardEdge(self, edge_index: np.ndarray):
        self.fA = sp.sparse.csr_matrix(
            (np.ones((edge_index.shape[1])),
             (edge_index[0],
              edge_index[1])), shape=(self.nV, self.nV))

    # 有向图中的边反向输入
    def setBackwardEdge(self, edge_index: np.ndarray):
        self.bA = sp.sparse.csr_matrix(
            (np.ones((edge_index.shape[1])),
             (edge_index[0],
              edge_index[1])), shape=(self.nV, self.nV))

    # def combine fA and bA and selfcircle
    def combineForwBack(self, fratio: float,
                        bratio: float, selfCircleRatio: float):
        self.A = fratio*self.fA+bratio*self.bA + \
            selfCircleRatio*sp.sparse.diags(np.ones((self.nV)))
        self.D = np.array(self.A.sum(axis=1)).flatten()
        self.Trev = sp.sparse.diags(1/self.D)*self.A

    def setDenseX(self, X: np.ndarray):
        self.rawX = X
        self.X = X

    def setSparseX(self, X: np.ndarray):
        self.rawX = sp.sparse.csr_matrix(X)
        self.X = X

    def reset(self):
        self.X = self.rawX
        self.nF = self.X.shape[1]

    def compressSparse(self, dim: int, pca_train_ratio: float):
        self.nF = dim
        self.pca = TruncatedSVD(n_components=dim, n_iter=10)
        self.pca.fit(divideArray(self.rawX, pca_train_ratio))
        self.X = self.pca.transform(self.rawX)
        return self.pca.explained_variance_ratio_

    def compressDense(self, dim: int, pca_train_ratio: float):
        self.nF = dim
        self.pca = PCA(n_components=dim, iterated_power=10)
        self.pca.fit(divideArray(self.rawX, pca_train_ratio))
        self.X = self.pca.transform(self.rawX)
        return self.pca.explained_variance_ratio_

    def precompute(self):
        self.Dnegr = np.power(self.D, -self.r).reshape(-1, 1)
        self.Q = np.zeros((self.L+1, self.nV, self.nF))
        np.multiply(self.Dnegr, self.X, out=self.Q[0])
        for i in range(self.L):
            self.Q[i+1] = self.Trev.dot(self.Q[i])
        self.getP(self.wl)

    def concatenateQ(self):
        return (1/self.Dnegr)*np.swapaxes(self.Q, 0, 1).reshape(self.nV, -1)

    # reset wl and get P
    def getP(self, wl):
        self.P = (1/self.Dnegr) * \
            np.tensordot(self.wl[::-1], self.Q, axes=[0, 0])
