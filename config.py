import os
import sys
import time

import torch
import torch.nn as nn

from models import *
from utils import *

device = "cuda" if torch.cuda.is_available() else "cpu"

model = BiLSTM(176, 256, 2, 3, 0.8, device).to(device)

loss_fn = nn.CrossEntropyLoss().to(device)

learning_rate = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 3000