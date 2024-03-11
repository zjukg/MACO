import dgl
import torch
import numpy as np
from dgl.nn import RelGraphConv

def build_kg(dataset):
    path = "./data/{}/train2id.txt".format(dataset)
    f = open(path, 'r')
    e1, e2, rels = [], [], []
    entity_map = {}
    for line in f.readlines()[1:]:
        h, t, r = line[:-1].split(' ')
        e1.append(int(h))
        e2.append(int(t))
        rels.append(int(r))
        entity_map[int(h)] = 1
        entity_map[int(t)] = 1
    for i in range(0, 14541):
        e1.append(i)
        e2.append(i)
        rels.append(237)
    graph = dgl.graph((e1, e2))
    return graph, rels
        
if __name__ == "__main__":
    graph, rel = build_kg('FB15K')
    rel = torch.tensor(rel)
    graph.ndata["x"] = torch.randn((graph.num_nodes(), 12))
    conv_layer = RelGraphConv(12, 12, 1345, regularizer='basis', num_bases=2)
    res = conv_layer(graph, graph.ndata["x"], etypes=rel)
    print(res.shape)

