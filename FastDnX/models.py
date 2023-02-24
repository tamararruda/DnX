import torch
from torch.nn import ReLU, Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, BatchNorm, GATConv, SGConv
from torch_geometric.nn import GATConv

class SGCNet(torch.nn.Module):
    def __init__(self, num_features, num_classes, hop):
        super(SGCNet, self).__init__()
        self.conv1 = SGConv(num_features, num_classes, hop)

    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        return input_lin

    def embedding(self, x, edge_index, edge_weights=None):
        out1 = self.conv1(x, edge_index, edge_weights)
        return out1

class GATNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=8, dropout=0.6)
        # On the Pubmed dataset, use heads=8 in conv2.
        self.conv2 = GATConv(8 * 8, dataset.num_classes, heads=1, concat=False,
                             dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GCNNet(torch.nn.Module):
    def __init__(self, dataset):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
        
        

class GCN(torch.nn.Module):
    """
    A graph clasification model for nodes decribed in https://arxiv.org/abs/2011.04573.
    This model consists of 3 stacked GCN layers and batch norm, followed by a linear layer.
    """
    def __init__(self, num_features, num_classes):
        super(GCN, self).__init__()
        self.embedding_size = 20 * 3
        self.conv1 = GCNConv(num_features, 20)
        self.relu1 = ReLU()
        self.bn1 = BatchNorm(20)        # BN is not used in GNNExplainer
        self.conv2 = GCNConv(20, 20)
        self.relu2 = ReLU()
        self.bn2 = BatchNorm(20)
        self.conv3 = GCNConv(20, 20)
        self.relu3 = ReLU()
        self.lin = Linear(self.embedding_size, num_classes)


    def forward(self, x, edge_index, edge_weights=None):
        input_lin = self.embedding(x, edge_index, edge_weights)
        out = self.lin(input_lin)
        return out

    def embedding(self, x, edge_index, edge_weights=None):
        stack = []

        out1 = self.conv1(x, edge_index, edge_weights)
        out1 = self.relu1(out1)
        out1 = self.bn1(out1)
        stack.append(out1)

        out2 = self.conv2(out1, edge_index, edge_weights)
        out2 = self.relu2(out2)
        out2 = self.bn2(out2)
        stack.append(out2)

        out3 = self.conv3(out2, edge_index, edge_weights)
        out3 = self.relu3(out3)
        stack.append(out3)

        input_lin = torch.cat(stack, dim=1)

        return input_lin
