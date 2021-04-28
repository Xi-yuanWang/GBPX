import numpy as np
import scipy as sp
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from utils.util import divideArray
from numba.typed import List
from numba import jit
from utils.PPRPush import EdgeList
# import time
# Dense Feature

class Graph:
    '''
    nV: int
    nF: int
    r: float
    wl: np.ndarray
    L: int
    X: np.ndarray
    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    EL: EdgeList
    rmax: np.ndarray
    '''

    def __init__(self, nV: int, nF: int, r: float, wl:
                 np.ndarray, L: int, rmax: float):
        self.nV = nV
        self.nF = nF
        self.r = r
        self.wl = wl
        self.L = L
        self.EL = EdgeList(nV, L)
        self.rmax = np.empty((nF), dtype=np.double)
        self.rmax.fill(rmax)
        self.delta = rmax

    def setEdge(self, edge_index: np.ndarray):
        self.A = sp.sparse.csr_matrix(
            (np.ones((edge_index.shape[1])),
             (edge_index[0],
              edge_index[1])), dtype=np.double, shape=(self.nV, self.nV))
        self.D = np.array(self.A.sum(axis=1)).flatten()
        sD = self.D + (self.D < 0.5)
        self.Drev = 1/sD
        self.Dnegr = np.power(sD, -self.r)
        self.Trev = sp.sparse.diags(self.Drev)*self.A
        self.EL.addEdge(edge_index[0], edge_index[1])
        self.T = None

    def addEdge(self, edge_index: np.ndarray):
        dA = sp.sparse.csr_matrix((np.ones(
            (edge_index.shape[1])), (edge_index[0], edge_index[1])), dtype=np.double, shape=(self.nV, self.nV))
        dD = sp.sparse.csr_matrix(dA.sum(axis=1))
        dDdat = dD.data
        dDrow = dD.nonzero()[0]
        # jit?
        D_trunc = self.D[dDrow]
        nD_trunc = D_trunc+dDrow
        self.D[dDrow] = nD_trunc
        snD_trunc = nD_trunc+(nD_trunc < 0.5)
        Drev_trunc = self.Drev[dDrow]
        nDrev_trunc = 1/snD_trunc
        self.Drev[dDrow] = nDrev_trunc
        Dnegr_trunc = self.Dnegr[dDrow]
        nDnegr_trunc = np.power(snD_trunc, -self.r)
        self.Dnegr[dDrow] = nDnegr_trunc

        dTrev = sp.sparse.csr_matrix(
            (-dDdat*Drev_trunc*nDrev_trunc, (dDrow, dDrow)),
            shape=(self.nV, self.nV))@self.A+sp.sparse.csr_matrix(
                (nDrev_trunc, (dDrow, dDrow)), shape=(self.nV, self.nV))@dA
        self.R[0] += sp.sparse.csr_matrix((nDnegr_trunc-Dnegr_trunc, (dDrow, dDrow)), shape=(
            self.nV, self.nV), dtype=np.double)@self.X
        for il in range(1, self.L+1):
            self.R[il] += dTrev@self.Q[il-1]
        self.Trev += dTrev
        self.A += dA
        self.EL.addEdge(edge_index[0], edge_index[1])
        self.EL.push(dDrow, self.Drev, self.Q, self.R, self.rmax)
        self.T = None

    def setDenseX(self, X: np.ndarray):
        self.rawX = X.astype(np.double)
        self.X = X.astype(np.double)

    def setSparseX(self, X: np.ndarray):
        self.rawX = sp.sparse.csr_matrix(X, dtype=np.double)
        self.X = X.astype(np.double)

    def reset(self):
        self.X = self.rawX
        self.nF = self.X.shape[1]

    def compressSparse(self, dim: int, pca_train_ratio: float):
        self.pca = TruncatedSVD(n_components=dim, n_iter=10)
        self.pca.fit(divideArray(self.rawX, pca_train_ratio))
        self.X = self.pca.transform(self.rawX)
        self.rmax = np.empty((dim), dtype=np.double)
        self.rmax.fill(self.nF/dim*self.delta)
        self.rmax /= np.max(
            np.stack((np.max(self.X, axis=0), -np.min(self.X, axis=0))).reshape(2, -1), axis=0)
        self.nF = dim
        return self.pca.explained_variance_ratio_

    def compressDense(self, dim: int, pca_train_ratio: float):
        self.pca = PCA(n_components=dim, iterated_power=14)
        self.pca.fit(divideArray(self.rawX, pca_train_ratio))
        self.X = self.pca.transform(self.rawX)
        self.rmax = np.empty((dim), dtype=np.double)
        self.rmax.fill(self.nF/dim*self.delta)
        self.rmax /= np.max(
            np.stack((np.max(self.X, axis=0), -np.min(self.X, axis=0))).reshape(2, -1), axis=0)
        self.nF = dim
        return self.pca.explained_variance_ratio_

    def precompute(self):
        self.Q = np.zeros((self.L+1, self.nV, self.nF), dtype=np.double)
        np.multiply(self.Dnegr.reshape(-1, 1), self.X, out=self.Q[0])
        for i in range(self.L):
            self.Q[i+1] = self.Trev.dot(self.Q[i])
        self.R = np.zeros((self.L+1, self.nV, self.nF), dtype=np.double)
        # self.getP(self.wl)

    def concatenateQ(self):
        return (1/self.Dnegr.reshape(-1, 1))*np.swapaxes(self.Q, 0, 1).reshape(self.nV, -1)

    def concatenateQForConv(self):
        return (1/self.Dnegr.reshape(-1, 1)).reshape(-1, 1, 1)*np.transpose(self.Q, (1, 2, 0))

    # warning: node with no edge
    # 整个放入c++。传入T
    def randomWalk(self, snode: np.ndarray, nr: int):
        if(self.T is not None):
            return self.T
        self.T = np.ascontiguousarray(
            self.Q[:, snode, :].copy().transpose(1, 2, 0))
        self.EL.randomWalk(snode, self.T, self.R, nr)
        self.T = (1/self.Dnegr[snode]).reshape(-1, 1, 1)*(self.T)
        return self.T
    # reset wl and get P

    def getP(self, wl):
        self.P = (1/self.Dnegr.reshape(-1, 1)) * \
            np.tensordot(self.wl[::-1], self.Q, axes=[0, 0])
