
<div align="center">

<img src="https://raw.githubusercontent.com/crybot/potatorch/main/docs/potatorch-banner.png" width="100%" role="img">

**PotaTorch is a lightweight PyTorch framework specifically designed to run on hardware with limited resources.**

______________________________________________________________________

<!-- [![PyPI Status](https://pepy.tech/badge/potatorch)](https://pepy.tech/project/potatorch) -->
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/potatorch)](https://pypi.org/project/potatorch/)
[![PyPI version](https://badge.fury.io/py/potatorch.svg)](https://badge.fury.io/py/potatorch)
![GitHub commit activity](https://img.shields.io/github/commit-activity/w/crybot/potatorch)
[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/crybot/potatorch/blob/main/LICENSE)

</div>

### Installation
PotaTorch is published on PyPI, you can install it through pip:
```bash
pip install potatorch
```

or you can install it from sources:
```bash
git clone --single-branch -b main https://github.com/crybot/potatorch
pip install -e potatorch
````
______________________________________________________________________

### Minimal Working Example
You can run the following example directly from `examples/mlp.py` if you already have pytorch installed, or you can run it with docker through the provided scripts:
```bash
./build.sh && ./run.sh
```

The example trains a feed forward network on a toy problem:
```python
import torch
from torch import nn

from potatorch.training import TrainingLoop, make_optimizer
from potatorch.callbacks import ProgressbarCallback
from torch.utils.data import TensorDataset

# Fix a seed for TrainingLoop to make non-deterministic operations such as
# shuffling reproducible
SEED = 42
device = 'cuda'

epochs = 100
lr = 1e-4

# Define your model as a pytorch Module
model = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), 
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, 1))

# Create your dataset as a torch.data.Dataset
dataset = TensorDataset(torch.arange(1000).view(1000, 1), torch.sin(torch.arange(1000)))

# Provide a loss function and an optimizer
loss_fn = torch.nn.MSELoss()
optimizer = make_optimizer(torch.optim.Adam, lr=lr)

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
        batch_size=None,
        shuffle=False,
        device=device,
        num_workers=0,
        seed=SEED,
        val_metrics={'l1': nn.L1Loss(), 'mse': nn.MSELoss()},
        callbacks=[
            ProgressbarCallback(epochs=epochs, width=20),
            ]
        )
# Run the training loop
model = training_loop.run(model, epochs=epochs)
```
______________________________________________________________________

### Automatic Hyperparameters Optimization
PotaTorch provides a basic set of utilities to perform hyperparameters optimization. You can choose among **grid search**, **random search** and **bayesian search**. All of them are provided by `potatorch.optimization.tuning.HyperOptimizer`. The following is a working example of a simple grid search on a toy problem. You can find the full script under `examples/grid_search.py`

```python
def train(dataset, device, config):
    """ Your usual training function that runs a TrainingLoop instance """
    SEED = 42
    # `epochs` is a fixed hyperparameter; it won't change among runs
    epochs = config['epochs']

    # Define your model as a pytorch Module
    model = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1))

    loss_fn = torch.nn.MSELoss()
    # `lr` is a dynamic hyperparameter; it will change among runs
    optimizer = make_optimizer(torch.optim.Adam, lr=config['lr'])

    training_loop = TrainingLoop(
            dataset,
            loss_fn,
            optimizer,
            train_p=0.8,
            val_p=0.1,
            test_p=0.1,
            random_split=False,
            batch_size=None,
            shuffle=False,
            device=device,
            num_workers=0,
            seed=SEED,
            val_metrics={'l1': nn.L1Loss(), 'mse': nn.MSELoss()},
            callbacks=[
                ProgressbarCallback(epochs=epochs, width=20),
                ]
            )
    model = training_loop.run(model, epochs=epochs, verbose=1)
    # Return a dictionary containing the training and validation metrics 
    # calculated during the last epoch of the loop
    return training_loop.get_last_metrics()

# Define your search configuration
search_config = {
        'method': 'grid',   # which search method to use: ['grid', 'bayes', 'random']
        'metric': {
            'name': 'val_loss', # the metric you're optimizing
            'goal': 'minimize'  # whether you want to minimize or maximize it
        },
        'parameters': { # the set of hyperparameters you want to optimize
            'lr': {
                'values': [1e-2, 1e-3, 1e-4]    # a range of values for the grid search to try
            }
        },
        'fixed': {      # fixed hyperparameters that won't change among runs
            'epochs': 200
        }
    }

def main():
    device = 'cuda'
    dataset = TensorDataset(torch.arange(1000).view(1000, 1), torch.sin(torch.arange(1000)))
    # Apply additional parameters to the train function to have f(config) -> {}
    score_function = partial(train, dataset, device)
    # Construct the hyperparameters optimizer
    hyperoptimizer = HyperOptimizer(search_config)
    # Run the optimization over the hyperparameters space
    config, error = hyperoptimizer.optimize(score_function, return_error=True)
    print('Best configuration found: {}\n with error: {}'.format(config, error))
```
