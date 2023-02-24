import numpy as np
from sklearn.metrics import roc_auc_score
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from torch_geometric.utils import k_hop_subgraph
import torch
    
def edge_to_node(values, edges):

    nodes_neight = np.unique(edges.flatten())

    node_expl =  np.zeros(nodes_neight.max()+1)
    node_count =  np.ones(nodes_neight.max()+1)

    for r, c, ex in zip(edges[0], edges[1], values):
        node_expl[r] += ex/2
        node_count[r]+=1
        node_expl[c] += ex/2
        node_count[c] += 1

    node_expl = node_expl/node_count

    return nodes_neight, node_expl[nodes_neight]
    
  
  
def node_to_edge(values, nodes, edges_neigh):
    nodes_idx, counts = edges_neigh.flatten().unique(return_counts=True)
    ex_node = {}
    for i, j in zip(nodes_idx.numpy(), values):
        ex_node[i] = j
    expl = []
    for edge in edges_neigh.T:
        i = edge[0].item() 
        j = edge[1].item()
        try:
            expl.append(ex_node[i]+ex_node[j])
        except:
            break

    return edges_neigh.numpy(), np.array(expl)
    
def get_ground_truth(node,dataset):
    gt = []
    if dataset == 'syn1':
        gt =  get_ground_truth_syn1(node) #correct
    elif dataset == 'syn2':
        gt =  get_ground_truth_syn1(node) #correct
    elif dataset == 'syn3':
        gt =  get_ground_truth_syn3(node) #correct
    elif dataset == 'syn4':
        gt =  get_ground_truth_syn4(node) #correct
    elif dataset == 'syn5':
        gt =  get_ground_truth_syn5(node) #correct
    elif dataset == 'syn6':
        gt =  get_ground_truth_syn1(node) #correct
    return gt

def get_ground_truth_syn1(node):
    base = [0,1,2,3,4]
    ground_truth = []
    offset = node % 5
    ground_truth = [node - offset + val for val in base]   
    return ground_truth

def get_ground_truth_syn3(node):
    base = [0,1,2,3,4,5,6,7,8]
    buff = node - 3
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 3 for val in base]   
    return ground_truth

def get_ground_truth_syn4(node):
    buff = node - 1
    base = [0,1,2,3,4,5]
    ground_truth = []
    offset = buff % 6
    ground_truth = [buff - offset + val + 1 for val in base]   
    return ground_truth

def get_ground_truth_syn5(node):
    base = [0,1,2,3,4,5,6,7,8]
    buff = node - 7
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 7 for val in base]   
    return ground_truth
    
def get_ground_truth_edge(edge_index, labels,dataset):

    labels_edge = []

    for i,j in edge_index.T:
        if labels[i.item()] == 0 or labels[j.item()] == 0:
            labels_edge.append(0)
        elif labels[i.item()] == 4 or labels[j.item()] == 4:
            if dataset == 'syn2':
                labels_edge.append(0)
            else:
                labels_edge.append(1)
        else:
            labels_edge.append(1)

    labels_edge = np.array(labels_edge)

    explanation_labels = (edge_index.numpy(), labels_edge)
    return explanation_labels
    
def evaluation_auc_node(explanations, explanation_labels):
    """Evaluate the auc score given explaination and ground truth labels.
    :param explanations: predicted labels.
    :param explanation_labels: ground truth labels.
    :returns: area under curve score.
    """
    ground_truth = []
    predictions = []
    for expl in explanations: # Loop over the explanations for each node
        ground_truth_node = []
        prediction_node = []

        for i in range(0, expl[0].shape[1]): # Loop over all edges in the explanation sub-graph
            prediction_node.append(expl[1][i]) #insere o peso da aresta na lista prediction_node

            # Graphs are defined bidirectional, so we need to retrieve both edges
            pair = expl[0].T[i]
            idx_edge = np.where((explanation_labels[0].T == pair).all(axis=1))[0]
            idx_edge_rev = np.where((explanation_labels[0].T == [pair[1], pair[0]]).all(axis=1))[0]

            # If any of the edges is in the ground truth set, the edge should be in the explanation
            gt = explanation_labels[1][idx_edge] + explanation_labels[1][idx_edge_rev]
            if gt == 0:
                ground_truth_node.append(0)
            else:
                ground_truth_node.append(1)

        ground_truth.extend(ground_truth_node)
        predictions.extend(prediction_node)

    score = roc_auc_score(ground_truth, predictions)
    return score
    
def evaluate_bitcoin_explanation(explanations, dataset, label_sgc):
    # Get predictions
    pred = label_sgc
    pred_label = [np.argmax(p).item() for p in pred]
    
    # Get ground truth
    filename_pos = os.path.join('/Generate_XA_Data/ground_truth_explanation/'+dataset,dataset+'_pos.csv')
    filename_neg = os.path.join('/Generate_XA_Data/ground_truth_explanation/'+dataset,dataset+'_neg.csv')
    df_pos = pd.read_csv(filename_pos, header=None, index_col=0, squeeze=True).to_dict()
    df_neg = pd.read_csv(filename_neg, header=None, index_col=0, squeeze=True).to_dict()
    
    # Evaluate
    pred_pos = 0
    true_pos = 0
    for node in explanations:
        
        gt = []
        if pred_label[node] == 0:
            buff_str = df_neg[node].replace('[','')
            buff_str = buff_str.replace(']','')
            gt = [int(s) for s in buff_str.split(',')]
        else:
            buff_str = df_pos[node].replace('[','')
            buff_str = buff_str.replace(']','')
            gt = [int(s) for s in buff_str.split(',')]
        ex = explanations[node]

        for e in ex:
            pred_pos = pred_pos + 1
            if e in gt:
                # print(e)
                true_pos = true_pos + 1
        
    precision = true_pos/pred_pos
    print(f'real: {true_pos}')
    print(f'precision: {pred_pos}')
    print("Explainer's precision is ", precision)
    
    
    
def fidelity_neg(model, node_list, edge_index, k_vizinhos_importantes, X, L, explanations):
    
    m = torch.nn.Softmax(dim=1)
    nest=0
    pos_ = []
    neg_ = []

    for node in tqdm(node_list):

        nodes = explanations[node][0]
        expls = explanations[node][1]

        edges_neigh = k_hop_subgraph(int(node), 3, edge_index)[1]

        if len(expls) > k_vizinhos_importantes:
            topex, topid = torch.topk(expls, k_vizinhos_importantes)
            nodes = nodes[topid]
            expls = topex

        neighborhood = edges_neigh.numpy().T

        important_edges = np.empty(shape=(0, 2), dtype='long')

        for edge in neighborhood:
            if edge[0] in nodes and edge[1] in nodes:
                important_edges = np.insert(important_edges, 0, edge, axis=0)

        pred = m(model(X,  torch.tensor(important_edges).T )).argmax(dim=1)[node]
        real = m(model(X, edges_neigh)).argmax(dim=1)[node]
        
        fidelidade_neg = int(real == L[node]) - int(pred == L[node])
        neg_.append(fidelidade_neg)

    fidelity_neg = np.sum(neg_) / len(node_list)

    print(f'\nfidelidade-: {fidelity_neg}')
