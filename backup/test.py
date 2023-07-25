#!/usr/bin/env python
# coding: utf-8
import torch
from torch import nn
import torch.nn.functional as F


class CNN(torch.nn.Module):
    def __init__(self, in_channels=16, num_classes=3):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=4, stride=1, padding=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=4, stride=1, padding=2),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(256, 64, kernel_size=4, stride=1, padding=2),
            nn.ReLU()
        )

        # Calculate the output size of the convolutional layers
        self.output_size = self._get_conv_output_size()

        self.lin1 = nn.Linear(self.output_size, 1024)
        self.lin2 = nn.Linear(1024, num_classes)

    def _get_conv_output_size(self):
        # Dummy input to calculate the output size of the convolutional layers
        dummy_input = torch.zeros(1, 16, 24000)
        x = self._forward_conv(dummy_input)
        return x.view(x.size(0), -1).size(1)

    def _forward_conv(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.lin2(x)
        return x

if __name__ == '__main__':
    model = CNN()
    input = torch.ones([32, 16, 24000])
    output = model(input)
    print(output.shape)