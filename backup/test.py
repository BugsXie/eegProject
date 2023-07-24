#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F


class GRU(nn.Module):
    def __init__(self):
        super(GRU, self).__init__()

        self.gru_layer = nn.GRU(
            input_size=176,
            hidden_size=64,
            num_layers=2,
            bias=True,
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, segment_length, no_feature)
        )

        self.out = nn.Linear(64, 3)

    def forward(self, x):
        r_out, (h_n, h_c) = self.gru_layer(x.float(), None)
        r_out = F.dropout(r_out, 0.3)
        test_output = self.out(r_out[:, -1, :]) # choose r_out at the last time step
        return test_output

if __name__ == '__main__':
    gru = GRU()
    input = torch.ones([1, 176])
    output = gru(input)
    print(output)