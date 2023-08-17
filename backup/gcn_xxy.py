import math

import torch

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class GraphConvolutionXXY(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolutionXXY, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, cuda=False):
        output = torch.zeros([input.shape[0], input.shape[1], self.out_features])
        if cuda:
            output = output.cuda()
        for i in range(input.shape[0]):
            output[i, :, :] = torch.mm(input[i, :, :], self.weight)
            adjT = adj[i, :, :]
            adjT = adjT.transpose(0, 1)
            output[i, :, :] = torch.spmm(adjT, output[i, :, :])
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
