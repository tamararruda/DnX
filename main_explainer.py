import torch.nn as nn

import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, add_self_loops, remove_self_loops, k_hop_subgraph

#torch.manual_seed(12)
#torch.random.manual_seed(12)


from utils import *
from model import *
from explainer import *

dataset_path = 'dataset/'
out_path = '/explanation/'
in_path = 'trained_distiller/'

dataset = 'syn6'
#save_dir = '/content/drive/MyDrive/PGM_Explainer/PGM_Node/Train_GNN_model/ckpt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load model
A_np, X_np = load_XA(dataset, datadir = dataset_path)
num_nodes = X_np.shape[0]
L = load_labels(dataset, datadir = dataset_path)
num_classes = max(L) + 1
ckpt_destilado = load_ckpt('SGC_train_'+dataset,datadir = in_path)
# #SGC_1_camada_one_hot

A = torch.tensor(A_np, dtype=torch.float32).to(device)
X = torch.tensor(X_np, dtype=torch.float32).to(device)

X = F.one_hot(torch.sum(A,1).type(torch.LongTensor)).type(torch.float32).to(device)
input_dim = X.shape[1]

edge_index,_ = dense_to_sparse(A)
edge_index = edge_index.to(device)

model = Net(3, input_dim, num_classes).to(device)
model.load_state_dict(ckpt_destilado["model_state"])
model.eval()

pred_model = model(X,edge_index).to(device)


def generating_explanations(node, explainer,edge_index, X,t, k):
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
    node_to_explain = [i for [i] in np.argwhere(np.sum(A_np,axis = 0) > 2)]
    np.random.shuffle(node_to_explain)
    node_list = node_to_explain
    k=3



nodes_explanations = {}
results = {}

t = 5
itr_no = 0
for node in node_list:
    nodes_explanations_aux = {}

    itr_no += 1
    print("_______ Nó {}______iteração {}_de {}__".format(node, itr_no, len(node_list)))
    lista_loss = []

    neighbors, sub_edge_index, node_idx_new, _ = k_hop_subgraph(int(node), 3, edge_index, relabel_nodes=True)
    sub_X = X[neighbors]

    node_idx_new, sub_edge_index, sub_X, neighbors = node_idx_new.to(device), sub_edge_index.to(device), sub_X.to(
        device), neighbors.to(device)
    explainer = Explainer(len(neighbors), node_idx_new).to(device)
    opt = torch.optim.Adam(explainer.parameters(), lr=0.1)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min',
                                                           factor=0.5, min_lr=1e-5,
                                                           patience=20,
                                                           verbose=True)

    loss = nn.MSELoss()
    # loss = nn.CrossEntropyLoss()

    for epoch in range(100):

        explainer.train()
        opt.zero_grad()

        expl, pred_ex = explainer(sub_X, sub_edge_index, model, t)

        l = loss(pred_ex[node_idx_new], pred_model[node])  # *expl.mean()#*0.1)

        lista_loss.append(l.item())
        l.backward(retain_graph=True)

        opt.step()
        scheduler.step(l)

        print(f'Epoch: {epoch:03d}, Loss: {l:.16f}')

        nodes_explanations_aux[node] = generating_explanations(node, explainer, edge_index, X, t, k)[node]
        acc, prec = evaluate_syn_explanation(nodes_explanations_aux, dataset)
        if acc == 1.0:
            break

    params = list(explainer.parameters())
    min_point = torch.norm(params[0].grad)
    results[node] = ['minimum point:{} | number of neighbors: {} | accuracy: {}'.format(min_point, len(neighbors), acc)]

    nodes_explanations[node] = generating_explanations(node, explainer, edge_index, X, t, k)[node]
    acc, prec = evaluate_syn_explanation(nodes_explanations, dataset)

    print("Accuracy: ", acc)
    print("Precision: ", prec)

    del explainer

with open('nodes_explanations_{}_{}.txt'.format(dataset, 5), 'w') as f:
    f.write("%s\n" % nodes_explanations)
