import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, k_hop_subgraph
from torch_geometric.nn.conv.gcn_conv import gcn_norm

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

def A_k_hop(A, hop):
    edge_index, weight_edge = dense_to_sparse(A) # pegando aretas da matriz de adjacência

    edge_index, edge_weight = gcn_norm(edge_index.long(), add_self_loops=True) # fazendo a normalização laplaciana das arestas

    n_A = torch.zeros(A.shape) # gerando uma matriz de zeros com as mesmas dimensões da matriz A

    for i, j in enumerate(edge_index.T): # atribuindo os pesos de cada aresta a nova matriz criada
        n_A[j[0].item()][j[1].item()] = edge_weight[i]

    A_pot = n_A
    for i in range(hop-1):
        A_pot = torch.matmul(n_A, A_pot) # realizando k-hop da matriz normalizada
    return A_pot


explanations = {}
edge_index, weight_edge = dense_to_sparse(A)

model = Net(3, input_dim, num_classes).to(device)
model.load_state_dict(ckpt_destilado["model_state"])
model.eval()
edge_index, _ = dense_to_sparse(A)

pred_model = model(X, edge_index).to(device)

# pred_model = torch.softmax(model(X,edge_index),1).to(device)
# label = ckpt_destilado['save_data']['label'].to(device)

W = ckpt_destilado['model_state']['conv.lin.weight'].to(device)
bias = (ckpt_destilado['model_state']['conv.lin.bias']).to(device)

A_pot = A_k_hop(A, 3).to(device)
accs = []
#y = A_pot @ X @ W.T + bias

for no_alvo in node_list:
    nodes_neigh, _, node_ex, _ = k_hop_subgraph(int(no_alvo), 3, edge_index)
    results = []
    idxs = []

    # multiplica cada node-feature pelo peso correspondete da matriz A_pot
    # for idx, Xi in zip(nodes_neigh, X[nodes_neigh]):
    #     S = Xi * A_pot[no_alvo, int(idx.item())]
    #     results.append(S.tolist())
    #     idxs.append(idx.tolist())

    results = (X[nodes_neigh].T * A_pot[no_alvo, nodes_neigh]).T
    results = results.to(device)
    expl = []

    list_pred = []
    for i in results:
        pred = torch.matmul(i, W.T)  # multiplica os pesos do SGC pelos vetores Xi ponderados

        L = pred_model[no_alvo] - bias
        projection = torch.matmul(pred.T, L)
        tam_projection = projection

        # tam_projection = torch.norm(pred)

        # validando
        # list_pred.append(pred.tolist())
        # torch.tensor(list_pred).sum(dim=0).to(device) + bias

        expl.append(tam_projection.tolist())
    expl = torch.tensor(expl)

    # dar mais importância ao nó alvo
    expl[torch.where(nodes_neigh == no_alvo)] = expl.sum()

    if len(nodes_neigh) < k:
        k = len(nodes_neigh)
    values, nodes = torch.topk(expl, dim=0, k=k)

    explanations[no_alvo] = nodes_neigh[nodes].tolist()

acc, prec = evaluate_syn_explanation(explanations,dataset)
print("Accuracy: ", acc)
print("Precision: ", prec)


with open('nodes_explanations_{}.txt'.format(dataset), 'w') as f:
    f.write("%s\n" % explanations)
