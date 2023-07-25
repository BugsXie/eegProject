import torch
import torch.nn as nn


class BiLSTM(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            num_layers,
            num_classes,
            dropout=0.5,
            device="cpu",
    ):
        super(BiLSTM, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        # self.ffn = nn.Linear(hidden_size * 2, hidden_size)
        self.batchNorm1d = nn.BatchNorm1d(10)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            self.device
        )
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            self.device
        )
        out, _ = self.lstm(x, (h0, c0))
        out = self.batchNorm1d(out)
        out = self.dropout(out)
        out = self.relu(out)
        out = self.fc(out[:, -1, :])
        return out
