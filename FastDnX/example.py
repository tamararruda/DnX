import torch
import time
from utils import load_dataset, load_model, plot_graph
from models import SGCNet
from explainer import FastDnX

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset_name = 'syn1'

data = load_dataset(dataset_name)
input_dim = data.x.shape[1]
num_classes = num_classes = max(data.y) + 1
hop = 3
model = SGCNet(input_dim, num_classes.item(), hop)
model = load_model(model, 'SGC', dataset_name, device)
model.eval()
labels = model(data.x, data.edge_index)
node_target = 300

start = time.time()
expl_model = FastDnX(model, data.x, 'node', hop, data.edge_index, labels)
expl_model.prepare()
nodes, values  = expl_model.explain(node_to_be_explain = node_target, size_expl = 5)
#plot_graph(node_target, nodes, data.edge_index.to('cpu'), hop, labels.to('cpu').argmax(dim=1).numpy())
end = time.time()

print(f'node to be explain: {node_target}')
print(f'expl nodes found: {nodes}')
print(f'expl weights found: {values.detach().numpy()}')
print(f'execution time: {round(end-start, 4)}')
