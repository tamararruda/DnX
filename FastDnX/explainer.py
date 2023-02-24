import torch
import torch.nn as nn
import numpy as  np
from torch_geometric.utils import k_hop_subgraph
from utils import A_k_hop

class FastDnX():
    def __init__(self, model_to_explain, features, task, hop, edge_index):
        self.features = features
        self.hop = hop
        self.edge_index = edge_index
        self.labels = model_to_explain(features, edge_index)
        self.num_nodes = len(features)
        self.A_pot = []
        self.params = []
        self.model_to_explain = model_to_explain

        if task == "node":
            print('finding top nodes...')
        elif task == "edge":
            print('finding top nodes...')
        else:
            pass


    def prepare(self, ):
        A = torch.zeros((self.num_nodes, self.num_nodes))
        r, c = self.edge_index
        A[r, c] = 1
        self.A_pot = A_k_hop(A, self.hop)

        self.params = []
        for param in self.model_to_explain.parameters():
            self.params.append(param)

    def explain(self, node_to_be_explain):
        nodes_neigh = k_hop_subgraph(node_to_be_explain, self.hop, self.edge_index)[0]
        
        labels_node_expl = torch.zeros(self.labels[nodes_neigh].shape)
        # print(labels_node_expl + self.labels[node_to_be_explain], self.params[1].shape)
        labels_node_expl = labels_node_expl + self.labels[node_to_be_explain] - self.params[1]
        
        X_pond = (self.features * self.A_pot[node_to_be_explain].view(-1,1))[nodes_neigh]
        a = torch.matmul(X_pond, self.params[0].T)
        expl = torch.diag(torch.matmul(a, labels_node_expl.T))
        
        expl[torch.where(nodes_neigh==node_to_be_explain)] = expl.sum()
        
        return nodes_neigh, expl.detach()
