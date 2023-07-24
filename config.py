import os
import sys
import time

import torch
import torch.nn as nn

from models import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BiLSTM(176, 256, 10, 3, 0.5, device).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)

learning_rate = 1e-2
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 100
