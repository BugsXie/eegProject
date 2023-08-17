import torch.nn as nn
import torch.nn.functional as F
from gcn_xxy import GraphConvolutionXXY


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nclass, channel, hidden_layer, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolutionXXY(nfeat, nhid)
        self.hidden_layer = hidden_layer if hidden_layer >= 1 else 1
        if hidden_layer >= 2:
            for i in range(2, hidden_layer+1):
                exec('self.gc{} = GraphConvolutionXXY(nhid, nhid)'.format(i))
        self.dropout = dropout
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(channel*nhid, nclass)

    def forward(self, x, adj, cuda=False):
        x = F.relu(self.gc1(x, adj, cuda))
        if self.hidden_layer >= 2:
            for i in range(2, self.hidden_layer+1):
                x = F.dropout(x, self.dropout, training=self.training)
                x = eval('F.relu(self.gc{}(x, adj, cuda))'.format(i))
        x = self.flatten(x)
        x = F.log_softmax(self.linear(x), dim=1)
        return x
