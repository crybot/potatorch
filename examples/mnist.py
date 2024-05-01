from __future__ import print_function
import math
import torch
from torch import nn
import numpy as np

from potatorch.training import TrainingLoop, make_optimizer, BabyStepSchedulerBase
from potatorch.callbacks import ProgressbarCallback
from torch.utils.data import TensorDataset
from potatorch.training import Curriculum
from torchvision import datasets, transforms
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

# Fix a seed for TrainingLoop to make non-deterministic operations such as
# shuffling reproducible
SEED = 42
device = 'cuda'

epochs = 100
lr = 1e-3

transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

dataset = datasets.MNIST('../data', train=True, download=True,
                   transform=transform)

# Provide a loss function and an optimizer
loss_fn = torch.nn.NLLLoss()
optimizer = make_optimizer(torch.optim.Adam, lr=lr)

model = Net()

# Construct a TrainingLoop object.
# TrainingLoop handles the initialization of dataloaders, dataset splitting,
# shuffling, mixed precision training, etc.
# You can provide callback handles through the `callbacks` argument.
training_loop = TrainingLoop(
        dataset,
        loss_fn,
        optimizer,
        train_p=0.8,
        val_p=0.1,
        test_p=0.1,
        random_split=False,
        batch_size=2**10,
        shuffle=True,
        device=device,
        num_workers=0,
        seed=SEED,
        callbacks=[
            ProgressbarCallback(epochs=epochs, width=20),
            ]
        )
# Run the training loop
model = training_loop.run(model, epochs=epochs)
