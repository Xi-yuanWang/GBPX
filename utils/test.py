import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import numpy as np
import scipy as sp
import sys
sys.path.append("..")
from utils import util, model, precomputeWithRNumba
X = np.load("../elliptic/X.npy")
Y = np.load("../elliptic/Y.npz")
labeled_node, label = Y.get("node"), Y.get("label")
label -= 1
T = np.load("../elliptic/year.npy")
dedge = np.load("../elliptic/edge.npy").transpose()
edge = np.stack(
    (np.concatenate((dedge[0], dedge[1])), np.concatenate((dedge[1], dedge[0]))))
nodeTcnt = np.load("../elliptic/nodeTcnt.npy")
edgeTcnt = np.load("../elliptic/edgeTCnt.npy")


def diedge2biedge(diedge):
    return np.stack((np.concatenate((diedge[0], diedge[1])), np.concatenate((diedge[1], diedge[0]))))


GA = precomputeWithRNumba.Graph(len(X), len(X[0]), 0.3, np.array(
    [0.15*(0.85**i) for i in range(8+1)]), 8, 1e-4)
GA.setDenseX(X)
GA.setEdge(diedge2biedge(dedge[:, :220000]))
GA.compressDense(10, 0.99)
GA.precompute()
GA.addEdge(diedge2biedge(dedge[:, 220000:]))
np.save("./test.npy",GA.R)
print(GA.rmax,np.max(GA.R))