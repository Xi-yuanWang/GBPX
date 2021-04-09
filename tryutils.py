import GBP_utils
import numpy as np
from torch_geometric.datasets import CitationFull
Citeseer=CitationFull("../data","Citeseer")

G=GBP_utils.Graph(Citeseer.data.num_nodes,Citeseer.num_features,4,0.4)
G.fromNumpy(Citeseer.data.edge_index[0],Citeseer.data.edge_index[1])
print("precomputestart")
G.preCompute(Citeseer.data.x,1e-5)
print("precompute end")
wl=np.array([0.15,0.1275,0.108375,0.09211875,0.0690890625])
ans=[]
for s in range(Citeseer.data.num_nodes):
    if(s%50==0):
        print(s)
    G.randomWalk(0,s)
    ans.append(G.getPVec(wl,s))
np.save("./Citeseer",np.array(ans))