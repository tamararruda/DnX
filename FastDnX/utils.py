import os
import torch
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import k_hop_subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import dense_to_sparse

def plot_graph(no_alvo, nodes, edge_index, hop, label):

    edge_hop = k_hop_subgraph(int(no_alvo), hop, edge_index)[1]

    expl_edges = []
    for j in edge_hop.T:
        if j[0].item() in nodes and j[1].item() in nodes and j[0].item() != j[1].item():
            expl_edges.append(j.numpy())

    plt.figure(3,figsize=(5,5)) 

    G=nx.Graph()
    G.add_edges_from(expl_edges)
    G.add_nodes_from(nodes)
    pos = nx.spring_layout(G)
    nx.draw(G, pos=pos, 
            arrowstyle='->', 
            arrowsize=20, 
            width=2,
            with_labels=False, 
            node_size=500, 
            node_color=label[nodes],
            edge_color='green')
    nx.draw_networkx_labels(G, pos=pos)

    # plt.show()
    plt.savefig(os.path.join('plots','Graph.png'), format='PNG')

def A_k_hop(A, hop):
    edge_index, weight_edge = dense_to_sparse(A)
    edge_index, edge_weight = gcn_norm(edge_index.long())
    n_A = torch.zeros(A.shape)

    r, c = edge_index
    n_A[r,c] = edge_weight
    
    return torch.matrix_power(n_A, hop)

def load_model(model, model_name, dataset_name, device):
    path_model = os.path.join('checkpoints', model_name+dataset_name+'.pth.tar')
    ckpt_model = torch.load(path_model, map_location=torch.device('cpu'))
    model.load_state_dict(ckpt_model["model_state"])

    return model

def load_dataset(dataset_name):
    path_dataset = os.path.join('datasets', 'dataset_'+dataset_name+'.pt')
    data = torch.load(path_dataset, map_location=torch.device('cpu'))
    return data

# data.x, data.y, data.edge_index, data.mask_train, data.mask_test, data.mask_val,