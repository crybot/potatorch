import pkbar # progress bar for pytorch
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from training import TrainingLoop

class TrainingCallback():
    """ Training Callback base class """
    def __init__(self):
        return

    def on_train_start(self, state):
        return

    def on_train_end(self, state):
        return

    def on_train_batch_start(self, state):
        return

    def on_train_batch_end(self, state):
        return

    def on_train_epoch_start(self, state):
        return

    def on_train_epoch_end(self, state):
        return

    def on_validation_batch_start(self, state):
        return

    def on_validation_batch_end(self, state):
        return

    def on_validation_start(self, state):
        return

    def on_validation_end(self, state):
        return

class ProgressbarCallback(TrainingCallback):
    kbar: pkbar.Kbar

    def __init__(self, epochs, width=20):
        super().__init__()
        self.epochs = epochs
        self.width = width

    def on_train_epoch_start(self, state):
        super().on_train_epoch_start(state)
        epoch = state.get_state('epoch')
        num_batches = state.get_state('batches')
        ################################### Initialization ########################################
        if epoch is not None and num_batches is not None:
            self.kbar = pkbar.Kbar(
                    target=num_batches,
                    num_epochs=self.epochs,
                    epoch=epoch,
                    width=self.width,
                    always_stateful=False,
                    stateful_metrics=['lr'])
        # By default, all metrics are averaged over time. If you don't want this behavior, you could either:
        # 1. Set always_stateful to True, or
        # 2. Set stateful_metrics=["loss", "rmse", "val_loss", "val_rmse"], Metrics in this list will be displayed as-is.
        # All others will be averaged by the progbar before display.
        ###########################################################################################
    
    def on_train_batch_end(self, state):
        super().on_train_batch_end(state)
        # TODO: use self.metrics and self.state to update the progress bar
        batch = state.get_state('batch')
        lr = state.get_state('lr')
        loss = state.get_last_metric('loss')
        if batch is not None and loss and lr:
            self.kbar.update(batch, values=[('loss', loss), ('lr', lr)])

    def on_validation_end(self, state):
        super().on_validation_end(state)
        # TODO: use self.metrics to update the progress bar
        val_loss = state.get_last_metric('val_loss')
        if val_loss is not None:
            self.kbar.add(1, values=[('val_loss', val_loss)])


class LRSchedulerCallback(TrainingCallback):
    def __init__(self, optimizer, warmup_steps=1000):
        super().__init__()
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.lr_warmup = LinearLR(self.optimizer, start_factor=0.001, total_iters=self.warmup_steps)
        self.lr_decay = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
    
    def on_train_start(self, state):
        super().on_train_start(state)
        state.update_state('lr', self.lr_warmup.get_last_lr()[0])

    def on_train_batch_end(self, state):
        super().on_train_batch_end(state)
        self.lr_warmup.step()
        state.update_state('lr', self.lr_warmup.get_last_lr()[0])

    def on_validation_end(self, state):
        super().on_validation_end(state)
        val_loss = state.get_last_metric('val_loss')
        if val_loss is not None:
            self.lr_decay.step(val_loss)
            state.update_state('lr', self.optimizer.param_groups[0]['lr'])
