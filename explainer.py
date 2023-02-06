import torch
import torch.nn as nn

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DnX(nn.Module):
    def __init__(self, n_input, idx_node):
        super(DnX, self).__init__()
        self.E = nn.Parameter(torch.randn(n_input, 1, dtype=torch.float32))
        self.idx_node = idx_node

    def forward(self, x, edge_index, model, t=5):
        weight_node = torch.softmax(self.E / t, 0)
        
        y_expl = model(weight_node * x, edge_index)

        return weight_node, y_expl
