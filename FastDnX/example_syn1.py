import torch
import time
from utils import load_dataset, load_model, plot_graph
from models import SGCNet, GCN
from explainer import FastDnX
import numpy as np
from torch_geometric.utils import k_hop_subgraph
from evaluation import get_ground_truth, node_to_edge, get_ground_truth_edge, evaluation_auc_node, fidelity_neg
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = 'syn1'

data = load_dataset(dataset_name)
input_dim = data.x.shape[1]
num_classes = num_classes = max(data.y) + 1
model = SGCNet(input_dim, num_classes.item(), 3)
model = load_model(model, 'SGC', dataset_name, device)
model.eval()
labels = model(data.x, data.edge_index)

node_list = list(range(300,700)) # for syn1
k_vizinhos_importantes = 5 # for syn1

explainer = FastDnX(model, data.x, 'node', 3, data.edge_index)
explainer.prepare()

inicio = time.time()
explanations = {}

for no_alvo in np.array(node_list):
    nodes, values  = explainer.explain(int(no_alvo))
    explanations[no_alvo] = [nodes, values]

fim = time.time()
print(f'Time: {fim-inicio}')

np.save('./explanations/fastdnx_'+dataset_name+'_gcn.npy', np.array([explanations]))



# **************************************************************************************************
# **************************************************************************************************
# **************************************************************************************************

explanations = np.load('./explanations/fastdnx_'+dataset_name+'_gcn.npy', allow_pickle=True)[0]

import torch.nn as nn
m = nn.Softmax(dim=1)

## evaluate synthetic - node livel
accs = []
for no_alvo in tqdm(np.array(node_list)):
    nodes, expl = explanations[no_alvo]
    if len(nodes) > k_vizinhos_importantes:
        value_expl, idx_expl = torch.topk(expl, dim=0,k=k_vizinhos_importantes)
        node_expl = nodes[idx_expl]
    else:
        node_expl = nodes
        value_expl = expl
    if m(labels).argmax(dim=1)[no_alvo] == data.y[no_alvo]:
        real = np.array(get_ground_truth(no_alvo, dataset_name))
        acc = len(list(filter(lambda x: x in real, node_expl.numpy()))) / len(node_expl)
        accs.append(acc)

print(f'accuracy node level: {np.mean(accs)}')

# **************************************************************************************************
# **************************************************************************************************
# **************************************************************************************************

## evaluate synthetic - edge livel

all_expl_nodes = []
for no_alvo in tqdm(np.array(node_list)):
    values = explanations[no_alvo][1].detach().numpy()
    nodes = explanations[no_alvo][0].numpy()
    edges_neigh = k_hop_subgraph(int(no_alvo), 3, data.edge_index)[1]

    all_expl_nodes.append(node_to_edge(values, nodes, edges_neigh))
    
explanation_labels = get_ground_truth_edge(data.edge_index, data.y, dataset_name)
auc_score = evaluation_auc_node(all_expl_nodes, explanation_labels)
print(f'auc edge level: {auc_score}')

# **************************************************************************************************
# **************************************************************************************************
# **************************************************************************************************

## fidelity-
ckpt = torch.load('./checkpoints/GCN_'+dataset_name+'.pth.tar', map_location=torch.device(device))

x = torch.ones((700, 10))
model_gcn = GCN(num_features=x.shape[1], num_classes=num_classes.item())
model_gcn.load_state_dict(ckpt["model_state"]) 
model_gcn = model_gcn.to(device)
fidelity_neg(model_gcn, node_list, data.edge_index, k_vizinhos_importantes, x, data.y, explanations)
