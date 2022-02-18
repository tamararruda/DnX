import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Explainer(nn.Module):
    def __init__(self, n_input, idx_node):
        super(Explainer, self).__init__()
        self.E = nn.Parameter(torch.randn(n_input, 1, dtype=torch.float32))
        self.idx_node = idx_node

    def forward(self, x, edge_index, model, t=5):
        weight_node = torch.softmax(self.E / t, 0)
        # weight_node = torch.sigmoid(self.E)
        # weight_node = torch.relu(self.E)
        # weight_node = self.E
        weight_node = torch.cat((weight_node[:self.idx_node],
                                 torch.tensor([[1]]).to(device),
                                 weight_node[self.idx_node + 1:]))

        y_expl = model(weight_node * x, edge_index)

        return weight_node, y_expl

