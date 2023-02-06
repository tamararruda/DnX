
import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.utils import dense_to_sparse, add_self_loops, remove_self_loops, k_hop_subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_XA(dataname, datadir = "XAL"):
    prefix = os.path.join(datadir,dataname)
    filename_A = prefix +"_A.npy"
    filename_X = prefix +"_X.npy"
    A = np.load(filename_A)
    X = np.load(filename_X)
    return A, X

def load_labels(dataname, datadir = "XAL"):
    prefix = os.path.join(datadir,dataname)
    filename_L = prefix +"_L.npy"
    L = np.load(filename_L)
    return L

def create_filename(save_dir, dataname, isbest=False, num_epochs=-1):
    filename = os.path.join(save_dir, dataname)


    if isbest:
        filename = os.path.join(filename, "best")
    elif num_epochs > 0:
        filename = os.path.join(filename, str(num_epochs))
    return filename + ".pth.tar"

def load_ckpt(dataname, datadir = "XAL", isbest=False):
    '''Load a pre-trained pytorch model from checkpoint.
    '''
    filename = create_filename(datadir, dataname, isbest)
    print(filename)
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        ckpt = torch.load(filename, map_location=torch.device('cpu') )
    else:
        print("Checkpoint does not exist!")
        print("Checked path -- {}".format(filename))
        print("Make sure you have provided the correct path!")
        print("You may have forgotten to train a model for this dataset.")
        print()
        print("To train one of the paper's models, run the following")
        print(">> python train.py --dataset=DATASET_NAME")
        print()
        raise Exception("File not found.")
    return ckpt


def mask(A):

  num_nodes = A.shape[0]
  num_train = int(num_nodes * 0.8)
  num_val = int(num_nodes * 0.9)

  idx = [i for i in range(num_nodes)]

  np.random.shuffle(idx)
  train_mask = idx[:num_train]
  val_mask = idx[num_train:num_val]
  test_mask = idx[num_val:]

  mask_train = np.zeros(num_nodes)
  mask_train[train_mask] = 1
  mask_train = torch.tensor(mask_train,dtype=torch.bool)

  mask_val = np.zeros(num_nodes)
  mask_val[val_mask] = 1
  mask_val = torch.tensor(mask_val,dtype=torch.bool)

  mask_test = np.zeros(num_nodes)
  mask_test[test_mask] = 1
  mask_test = torch.tensor(mask_test,dtype=torch.bool)

  return mask_train, mask_val, mask_test



def get_nodes_explained(dataset, A):

    if dataset == 'syn1':
        node_list = list(range(300,700))
        k = 5
    elif dataset == 'syn2':
        node_list = list(range(300,700)) + list(range(1000,1400))
        k = 5
    elif dataset == 'syn3':
        node_list = list(range(300,1020))
        k=9
    elif dataset == 'syn4':
        node_list = list(range(511,871))
        k=6
    elif dataset == 'syn5':
        node_list = list(range(511,1231))
        k=9
    elif dataset == 'syn6':
        node_list = list(range(300,700))
        k=5
    else:
        node_to_explain = [i for [i] in np.argwhere(np.sum(A,axis = 0) > 2)]
        np.random.shuffle(node_to_explain)
        node_list = node_to_explain
        k=3

    return node_list, k

def create_filename_save(save_dir,dataset, isbest=False, num_epochs=-1):
    filename = os.path.join(save_dir, '')
    if isbest:
        filename = os.path.join(filename, "best")
    return filename +'SGC_'+dataset + ".pth.tar"


def save_checkpoint(model, optimizer, save_dir, dataset, num_epochs=100, isbest=False, save_data=None):
    filename = create_filename_save(save_dir, dataset, isbest, num_epochs=num_epochs)
    print(filename)
    torch.save(
        {
            "epoch": num_epochs,
            "optimizer": optimizer,
            "model_state": model,
            "optimizer_state": optimizer.state_dict(),
            "save_data": save_data,
        },
        str(filename),
    )


def evaluate_bitcoin_explanation(explanations, dataset, pred):
    # Get predictions
    # ckpt = load_ckpt(prog_args)
    # pred = ckpt["save_data"]["pred"]
    pred_label = np.argmax(pred, 1)

    # Get ground truth
    filename_pos = os.path.join(
        'dataset/' + dataset,
        dataset + '_pos.csv')
    filename_neg = os.path.join(
        'dataset/' + dataset,
        dataset + '_neg.csv')
    df_pos = pd.read_csv(filename_pos, header=None, index_col=0, squeeze=True).to_dict()
    df_neg = pd.read_csv(filename_neg, header=None, index_col=0, squeeze=True).to_dict()

    # Evaluate
    pred_pos = 0
    true_pos = 0
    for node in explanations:

        gt = []
        if pred_label[node] == 0:
            buff_str = df_neg[node].replace('[', '')
            buff_str = buff_str.replace(']', '')
            gt = [int(s) for s in buff_str.split(',')]
        else:
            buff_str = df_pos[node].replace('[', '')
            buff_str = buff_str.replace(']', '')
            gt = [int(s) for s in buff_str.split(',')]
        ex = explanations[node]

        for e in ex:
            pred_pos = pred_pos + 1
            if e in gt:
                true_pos = true_pos + 1
    precision = true_pos / pred_pos
    print("Explainer's precision is ", precision)
    return precision


def evaluate_syn_explanation(explanations, dataset):
    gt_positive = 0
    true_positive = 0
    pred_positive = 0
    for node in explanations:
        ground_truth = get_ground_truth(node, dataset)
        gt_positive = gt_positive + len(ground_truth)
        pred_positive = pred_positive + len(explanations[node])
        for ex_node in explanations[node]:
            if ex_node in ground_truth:
                true_positive = true_positive + 1

    accuracy = true_positive / gt_positive
    precision = true_positive / pred_positive

    return accuracy, precision


def get_ground_truth(node, dataset):
    gt = []
    if dataset == 'syn1':
        gt = get_ground_truth_syn1(node)  # correct
    elif dataset == 'syn2':
        gt = get_ground_truth_syn1(node)  # correct
    elif dataset == 'syn3':
        gt = get_ground_truth_syn3(node)  # correct
    elif dataset == 'syn4':
        gt = get_ground_truth_syn4(node)  # correct
    elif dataset == 'syn5':
        gt = get_ground_truth_syn5(node)  # correct
    elif dataset == 'syn6':
        gt = get_ground_truth_syn1(node)  # correct
    return gt


def get_ground_truth_syn1(node):
    base = [0, 1, 2, 3, 4]
    ground_truth = []
    offset = node % 5
    ground_truth = [node - offset + val for val in base]
    return ground_truth


def get_ground_truth_syn3(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 3
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 3 for val in base]
    return ground_truth


def get_ground_truth_syn4(node):
    buff = node - 1
    base = [0, 1, 2, 3, 4, 5]
    ground_truth = []
    offset = buff % 6
    ground_truth = [buff - offset + val + 1 for val in base]
    return ground_truth


def get_ground_truth_syn5(node):
    base = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    buff = node - 7
    ground_truth = []
    offset = buff % 9
    ground_truth = [buff - offset + val + 7 for val in base]
    return ground_truth

def A_k_hop(A, hop):
    edge_index, weight_edge = dense_to_sparse(A) 

    edge_index, edge_weight = gcn_norm(edge_index.long(), add_self_loops=True) 

    n_A = torch.zeros(A.shape) 

    for i, j in enumerate(edge_index.T): 
        n_A[j[0].item()][j[1].item()] = edge_weight[i]

    A_pot = n_A
    for i in range(hop-1):
        A_pot = torch.matmul(n_A, A_pot) 
    return A_pot


def generating_explanations(node,model, explainer,edge_index, X,t, k):
    explanations= {}
    explainer.eval()
    neighbors, sub_edge_index, node_idx_new, _ = k_hop_subgraph(int(node), 3, edge_index,relabel_nodes=True)
    sub_X = X[neighbors,:]
    node_idx_new, sub_edge_index, sub_X, neighbors =node_idx_new.to(device), sub_edge_index.to(device), sub_X.to(device), neighbors.to(device)

    expl, _ = explainer(sub_X,sub_edge_index,model,t)
    if len(neighbors)<k:
        k= len(neighbors)
    values, nodes = torch.topk(expl.squeeze(-1),dim=-1,k=k)
    explic = neighbors[nodes]
    explanations[node] =explic.tolist()
    
    return  explanations
