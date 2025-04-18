import torch
from torch import nn
import torch.nn.functional as F
from potatorch.training import TrainingLoop
from potatorch.callbacks import ProgressbarCallback
from torch.utils.data import TensorDataset

# Fix a seed for TrainingLoop to make non-deterministic operations such as
# shuffling reproducible
SEED = 42
device = 'cuda'

N = 100000
epochs = 100
lr = 1e-4

# Define your model as a pytorch Module
model = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), 
                      nn.Linear(128, 128), nn.ReLU(),
                      nn.Linear(128, 128), nn.ReLU(),
                      nn.Linear(128, 1)).to(device)

# Create your dataset as a torch.data.Dataset
dataset = TensorDataset(torch.arange(N).view(N, 1), torch.sin(torch.arange(N).view(N, 1)))

# Provide a loss function and an optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# Provide evaluation metrics. TrainingLoop expects a function of two arguments (pred, inputs): pred is the prediction of
# the model and inputs is a tuple containing both the input features and the target. Here we only need the target, so we
# discard the first entry of inputs
l1 = lambda pred, inputs: F.l1_loss(pred, inputs[1])
mse = lambda pred, inputs: F.mse_loss(pred, inputs[1])

# Construct a TrainingLoop object.
# TrainingLoop handles the initialization of dataloaders, dataset splitting,
# shuffling, mixed precision training, etc.
# You can provide callback handles through the `callbacks` argument.
training_loop = TrainingLoop(
    model,
    dataset,
    loss_fn,
    optimizer,
    train_p=0.8,
    val_p=0.1,
    test_p=0.1,
    random_split=False,
    batch_size=128,
    shuffle=False,
    device=device,
    num_workers=0,
    seed=SEED,
    val_metrics={'l1': l1, 'mse': mse},
    callbacks=[
        ProgressbarCallback(epochs=epochs, width=20),
    ]
)
# Run the training loop
model = training_loop.run(epochs=epochs)
