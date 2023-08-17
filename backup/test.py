#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Linear
import torch
from torch_geometric.nn import GATConv, global_mean_pool
import torch.nn.functional as F
from torcheeg.models import DGCNN

import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

class GNN(torch.nn.Module):
    def __init__(self, in_channels=4, num_layers=3, hid_channels=64, num_classes=3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hid_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hid_channels, hid_channels))
        self.lin1 = Linear(hid_channels, hid_channels)
        self.lin2 = Linear(hid_channels, num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x, edge_index, batch):
        # x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return x


if __name__ == '__main__':
    model = DGCNN(in_channels=40, num_electrodes=16, hid_channels=16, num_layers=2, num_classes=3)
    input = torch.ones([1, 16, 40])
    output = model(input)
    print('model: \n', model)
    print('output_shape: \n', output.shape)
    writer = SummaryWriter(r".\DGCNN_16_2")
    writer.add_graph(model, input)
    writer.close()