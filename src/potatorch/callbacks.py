import os
import pkbar # progress bar for pytorch
import wandb
import torch
import numpy as np
from torch import nn
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.optim.lr_scheduler import CosineAnnealingLR
from abc import ABC, abstractmethod
from potatorch.training import TrainingLoop

class TrainingCallback(ABC):
    """ Training Callback base class """
    def __init__(self):
        return
    
    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass

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

    def on_evaluation_end(self, state):
        return

class ProgressbarCallback(TrainingCallback):
    kbar: pkbar.Kbar

    def __init__(self, epochs, width=20):
        super().__init__()
        self.epochs = epochs
        self.width = width

    def state_dict(self):
        return None

    def load_state_dict(self, state_dict):
        pass

    def on_train_epoch_start(self, state):
        super().on_train_epoch_start(state)
        # reset incremental averages
        self.loss = 0.0
        self.lr = 0.0

        # pkbar expects epochs starting from zero
        epoch = state.get_state('epoch', 1) - 1
        num_batches = state.get_state('batches')
        verbose = state.get_state('verbose')
        ################################### Initialization ########################################
        if verbose and epoch is not None and num_batches is not None:
            self.kbar = pkbar.Kbar(
                    target=num_batches,
                    num_epochs=self.epochs,
                    epoch=epoch,
                    width=self.width,
                    always_stateful=True)
        # By default, all metrics are averaged over time. If you don't want this behavior, you could either:
        # 1. Set always_stateful to True, or
        # 2. Set stateful_metrics=["loss", "rmse", "val_loss", "val_rmse"], Metrics in this list will be displayed as-is.
        # All others will be averaged by the progbar before display.
        ###########################################################################################
    
    def on_train_batch_end(self, state):
        super().on_train_batch_end(state)
        # TODO: use self.metrics and self.state to update the progress bar
        batch = state.get_state('batch')
        verbose = state.get_state('verbose')
        self.lr = self.lr + (state.get_state('lr') - self.lr) / (batch + 1)
        self.loss = self.loss + (state.get_last_metric('loss') - self.loss) / (batch + 1)
        if verbose and batch is not None and self.loss and self.lr:
            self.kbar.update(batch, values=[('loss', self.loss), ('lr', self.lr)])

    def on_validation_end(self, state):
        super().on_validation_end(state)
        # TODO: use self.metrics to update the progress bar
        val_loss = state.get_last_metric('val_loss')
        verbose = state.get_state('verbose')
        other_metrics = state.val_metrics.keys()
        if verbose and val_loss is not None:
            # formatting rule for validation metrics: val_{metric}
            val_m = lambda m: f'val_{m}'
            self.kbar.add(1, values=[('val_loss', val_loss), 
                # retrieve and zip other validation metrics from the state
                *[(val_m(m), state.get_last_metric(val_m(m))) for m in other_metrics]
                ])

class LRSchedulerCallback(TrainingCallback):
    def __init__(self, optimizer, warmup_steps=1000, cosine_annealing=True, restart=False, cosine_tmax=None, cosine_factor=None, min_lr=0.0, config={}):
        super().__init__()
        # self.config = config
        self.optimizer = optimizer
        self.warmup_steps=config.get('warmup_steps', warmup_steps)
        self.lr_warmup = LinearLR(self.optimizer, start_factor=0.001, total_iters=self.warmup_steps)
        self.lr_decay = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.lr_cosine = None

        self.cosine_annealing = config.get('cosine_annealing', cosine_annealing)
        self.cosine_tmax = config.get('cosine_tmax', cosine_tmax)
        self.cosine_factor = config.get('cosine_factor', cosine_factor)
        self.restart = config.get('cosine_restart', restart)
        self.min_lr = config.get('min_lr', min_lr)

        if self.cosine_tmax is None and self.cosine_annealing:
            self.cosine_tmax = 50
            if not self.cosine_factor:
                self.cosine_factor = 1
            self.cosine_factor = int(self.cosine_factor)

    def reset(self):
        self.lr_warmup = LinearLR(self.optimizer, start_factor=0.001, total_iters=self.warmup_steps)
        self.lr_decay = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=5)
        self.lr_cosine = None # it will be initialized next epoch

    def _init_cosine_annealing(self):
        # With restarts
        if self.restart:
            self.lr_cosine = CosineAnnealingWarmRestarts(self.optimizer, self.cosine_tmax, self.cosine_factor, eta_min=self.min_lr)
        # Without restarts
        else:
            self.lr_cosine = CosineAnnealingLR(self.optimizer, self.cosine_tmax, eta_min=self.min_lr)

    def state_dict(self):
        state_dict = {}
        state_dict.update({
            'lr_warmup_state_dict': self.lr_warmup.state_dict(),
            'lr_decay_state_dict': self.lr_decay.state_dict()
            })

        if self.lr_cosine:
            state_dict.update({
                'lr_cosine_state_dict': self.lr_cosine.state_dict()
                })

        return state_dict

    def load_state_dict(self, state_dict):
        self.lr_warmup.load_state_dict(state_dict['lr_warmup_state_dict'])
        self.lr_decay.load_state_dict(state_dict['lr_decay_state_dict'])

        if self.cosine_annealing:
            if self.lr_cosine is None:
                self._init_cosine_annealing()
            self.lr_cosine.load_state_dict(state_dict.get('lr_cosine_state_dict', {}))
    
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

        # Cosine annealing
        if self.cosine_annealing:
            batches = state.get_state('batches')
            epoch = state.get_state('epoch')
            # Only when warmup is over
            if epoch is not None and batches and epoch * batches > self.warmup_steps:
                # Init cosine annealing if None
                if self.lr_cosine is None:
                    self._init_cosine_annealing()

                # Apply cosine annealing
                self.lr_cosine.step()
        # Decay on plateau (if cosine_annealing is False)
        elif val_loss is not None:
            self.lr_decay.step(val_loss)
        state.update_state('lr', self.optimizer.param_groups[0]['lr'])



class WandbCallback(TrainingCallback):
    def __init__(
            self,
            project_name,
            entity,
            config=None,
            tags=None,
            save_code=True,
            log=True,
            batch_frequency=None
            ):
        super().__init__()
        self.project_name = project_name
        self.entity = entity
        self.tags = tags
        self.config = config
        self.save_code = save_code
        self.log = log
        self.batch_frequency = batch_frequency
        self.run_id = None

    def state_dict(self):
        state_dict = {}
        state_dict.update({'run_id': self.run_id})
        return state_dict

    def load_state_dict(self, state_dict):
        self.run_id = state_dict.get('run_id', None)

    def init(self):
        run = wandb.init(
                id=self.run_id,
                project=self.project_name,
                entity=self.entity,
                config=self.config,
                resume='must' if self.run_id else None,
                tags=self.tags,
                save_code=self.save_code,
                reinit=True)
        if self.save_code:
            run.log_code(".")
        return run


    def on_train_start(self, state):
        super().on_train_start(state)
        # Create a new wandb run
        self.run = self.init()
        self.run_id = self.run.id

    def on_train_end(self, state):
        super().on_train_end(state)
        # Stop the current run
        if self.run:
            self.run.finish()

    def on_train_epoch_end(self, state):
        super().on_train_epoch_end(state)
        # We can probably do everything after validation
        # if self.log:
        #     wandb.log(state.get_states() | state.get_last_metrics(), commit=True)

    def on_train_batch_end(self, state):
        super().on_train_batch_end(state)
        batch = state.get_state('batch')
        if self.log and self.batch_frequency and batch \
                    and batch % self.batch_frequency == 0:
            wandb.log({**state.get_states(), **state.get_last_metrics()}, commit=True)

    def on_validation_end(self, state):
        super().on_validation_end(state)
        if self.log:
            wandb.log({**state.get_states(), **state.get_last_metrics()}, commit=True)

    def on_evaluation_end(self, state):
        super().on_evaluation_end(state)
        if self.log:
            wandb.log({**state.get_states(), **state.get_last_metrics()}, commit=True)


class CheckpointCallback(TrainingCallback):
    def __init__(self, path, save_best=True, detect_anomaly=False, metric='val_loss', frequency=None, mode='min', sync_wandb=False, debug=False):
        super().__init__()
        self.path = path
        self.save_best = save_best
        self.detect_anomaly = detect_anomaly
        self.metric = metric
        self.frequency = frequency
        self.mode = mode
        self.sync_wandb = sync_wandb
        self.debug = debug
        self.anomaly_detected = False

        assert path, 'path must not be None'
        assert save_best or frequency or detect_anomaly, 'Either one among save_best, frequency and detect_anomaly must be provided'
        assert not (save_best and frequency), 'Only one among save_best, frequency and detect_anomaly can be provided'
        assert not (save_best and detect_anomaly), 'Only one among save_best, frequency and detect_anomaly can be provided'
        assert not (frequency and detect_anomaly), 'Only one among save_best, frequency and detect_anomaly can be provided'
        assert (not save_best) or (mode == 'min' or mode == 'max'), "If save_best is True, then mode can either be 'min' or 'max'"
        assert (not save_best) or metric, 'If save_best = True, then a metric must be provided'

        if self.save_best:
            self.best = np.inf if mode == 'min' else -np.inf
            self.default = self.best

        if self.save_best and self.mode:
            self.minimize = mode == 'min'

    def state_dict(self):
        state_dict = {}
        if self.save_best:
            state_dict.update({'best': self.best})
        return state_dict

    def load_state_dict(self, state_dict):
        if 'best' in state_dict:
            self.best = state_dict['best']

    def on_train_start(self, state):
        super().on_train_start(state)

    def on_train_end(self, state):
        super().on_train_end(state)

    def on_train_epoch_end(self, state):
        super().on_train_epoch_end(state)
        epoch = state.get_state('epoch', 1)
        output = False

        # Save model checkpoint after each epoch, until an anomaly is detected
        if self.detect_anomaly and not self.anomaly_detected:
            metric = state.get_last_metric(self.metric)
            if torch.isnan(metric) or torch.isinf(metric):
                self.anomaly_detected = True
                print('Anomaly detected, disabling model checkpointing')
            else:
                self._save_checkpoint(state)
                output = True

        if self.frequency and epoch % self.frequency == 0:
            self._save_checkpoint(state)
            output = True

        if self.save_best:
            current = state.get_last_metric(self.metric, self.default)
            if (self.minimize and current < self.best) or \
               (not self.minimize and current > self.best):
                   self.best = current
                   self._save_checkpoint(state)
                   output = True
        if self.debug and output:
            print('Saving model checkpoint')

    def _save_checkpoint(self, state):
        """ Save the current state checkpoint to the specified self.path """
        # Save dump to disk
        torch.save(state.dump_state(), self.path)
        # Sync file with wandb
        if self.sync_wandb:
            wandb.save(self.path, base_path=os.path.dirname(self.path))

class SanityCheckCallback(TrainingCallback):
    def __init__(self, data, descriptors=None, transform=None, target_transform=None):
        self.data = data
        self.descriptors = descriptors
        self.transform = transform
        self.target_transform = target_transform
        self.data = self._make_tensors(self.data)

    def _make_tensors(self, data):
        return [tuple(torch.tensor(x) for x in t) for t in data]

    def state_dict(self):
        return {}

    def load_state_dict(self, state_dict):
        pass

    def on_train_epoch_end(self, state):
        h = state.predict(self.data)
        if self.descriptors is not None:
            samples = zip(self.descriptors, self.data, h)
        else:
            samples = zip(range(len(self.data)), self.data, h)

        # fixed column widths # TODO: constructor arguments
        W1, W2, W3 = 70, 20, 20

        # header
        header = f"{'Descriptor':<{W1}} {'Prediction':>{W2}} {'Target':>{W3}}"
        print(header)
        print("-" * len(header))
        for (i, x, h) in samples:
            y = x[-1] # The last value in the input tensor correspond to the supervised target
            if self.target_transform:
                h = self.target_transform(h)
            if hasattr(y, "item"):
                y = y.item()
            if hasattr(h, "item"):
                h = h.item()
            print(
                f"{str(i):<{W1}} "
                    f"{str(h):>{W2}} "
                    f"{str(y):>{W3}}"
            )
