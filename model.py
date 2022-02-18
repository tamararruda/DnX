
import torch
from torch_geometric.nn import  SGConv


class Net(torch.nn.Module):
    def __init__(self,k, nfeat, nclass):
          super(Net, self).__init__()
          self.conv = SGConv(nfeat, nclass, k)

    def forward(self, x, edge_index):

          x = self.conv(x, edge_index)

          return x
