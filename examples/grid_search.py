import torch
from torch import nn

from potatorch.training import TrainingLoop, make_optimizer
from potatorch.callbacks import ProgressbarCallback
from potatorch.optimization.tuning import HyperOptimizer
from torch.utils.data import TensorDataset
from functools import partial

def train(dataset, device, config):
    """ Your usual training function that runs a TrainingLoop instance """
    # Fix a seed for TrainingLoop to make non-deterministic operations such as
    # shuffling reproducible
    SEED = 42
    # NOTE: `epochs` is a fixed hyperparameter; it won't change among runs
    epochs = config['epochs']

    # Define your model as a pytorch Module
    model = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), 
            nn.Linear(128, 128), nn.ReLU(),
            nn.Linear(128, 1))

    loss_fn = torch.nn.MSELoss()
    # Provide a loss function and an optimizer
    # NOTE: `lr` is a dynamic hyperparameter; it will change among runs
    optimizer = make_optimizer(torch.optim.Adam, lr=config['lr'])

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
    # Create your dataset as a torch.data.Dataset
    dataset = TensorDataset(torch.arange(1000).view(1000, 1), torch.sin(torch.arange(1000)))
    # Apply additional parameters to the train function to have f(config) -> {}
    score_function = partial(train, dataset, device)
    # Construct the hyperparameters optimizer
    hyperoptimizer = HyperOptimizer(search_config)
    # Run the optimization over the hyperparameters space
    config, error = hyperoptimizer.optimize(score_function, return_error=True)
    print('Best configuration found: {}\n with error: {}'.format(config, error))

if __name__ == '__main__':
    main()
