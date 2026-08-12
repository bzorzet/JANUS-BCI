import numpy as np
import torch
import time
import os
import copy

class Callback:
    def on_train_begin(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_batch_begin(self, trainer, batch_idx): pass
    def on_batch_end(self, trainer, batch_idx): pass
    def on_epoch_end(self, trainer, epoch): pass
    def on_train_end(self, trainer): pass

#%%################################################# SIMPLE CALLBACKS ####################################################
class TimerCallback():
    def on_train_begin(self, trainer):
        self.start_time = time.time()

    def on_train_end(self, trainer):
        elapsed_time = time.time() - self.start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        trainer.history['duration'] = elapsed_time

class LossTracker():
    def __init__(self, name='train_loss', on_training=True):
        self.name = name  # Either 'train' or 'val'
        self.on_training = on_training  # Whether this loss is for training or validation
        self.reset()

    def reset(self):
        self.batch_losses = []
        self.epoch_losses = []

    def on_train_begin(self, trainer):
        self.reset()
        if self.name not in trainer.history:
            trainer.history[self.name] = []

    def on_epoch_begin(self, trainer, epoch, **kwargs):
        self.batch_losses = []

    def on_batch_end(self, trainer, batch_idx, loss=None, training=True,**kwargs):
        if self.on_training == training:
            if loss is not None:
                self.batch_losses.append(loss)

    def on_epoch_end(self, trainer, epoch, **kwargs):
        if self.batch_losses:
            epoch_loss = sum(self.batch_losses) / len(self.batch_losses)
            trainer.history[self.name].append(epoch_loss)
            if self.on_training:
                trainer.history['epoch'].append(epoch)

    def on_train_end(self, trainer, **kwargs):
        if not self.on_training and 'val_loss' in trainer.history:
            trainer.history['best_epoch'] = trainer.history['epoch'][np.argmin(trainer.history['val_loss'])]
        else:
            trainer.history['best_epoch'] = trainer.history['epoch'][np.argmax(trainer.history['train_loss'])]

class EpochScoring():
    def __init__(self, score_fn, name, on_training=True):

        self.score_fn = score_fn        # scoring function must be a function that can handle y_pred as probabilities
        self.name = name                # name to track in history
        self.on_training = on_training # whether this scoring is for training or validation
        self.scores = []

        self.y_pred = []
        self.y_true = []

    def on_train_begin(self, trainer):
        if self.name not in trainer.history:
            trainer.history[self.name] = []

    def on_epoch_begin(self, trainer, epoch, **kwargs):
        self.y_pred = []
        self.y_true = []

    def on_batch_end(self, trainer, batch_idx, y_pred=None, y_true=None, training=True, **kwargs):
        if self.on_training == training:
            if y_pred is not None and y_true is not None:
                self.y_pred.append(y_pred.detach().cpu())
                self.y_true.append(y_true.detach().cpu())

    def on_epoch_end(self, trainer, epoch, **kwargs):
        if self.y_pred and self.y_true:
            all_preds = torch.cat(self.y_pred)
            all_trues = torch.cat(self.y_true)
            score = self.score_fn(all_preds, all_trues)
            # Save in the history training or validation
            trainer.history[self.name].append(score)

#%% ################################################# CHECKPOINTS ####################################################
class CustomCheckpoint():
    def __init__(self, dirname, save_each_n_epochs=None, save_best=True, **kwargs):
        # Always we save the first epoch and the last epoch
        self.dirname = dirname
        self.save_each_n_epochs = save_each_n_epochs
        # NOTE: renamed from `save_best_model` -> `save_best` to match the
        # actual key every JSON config uses (`callbacks[].params.save_best`).
        # In repo_viejo this was `save_best_model`, so every config's
        # `save_best` fell silently into **kwargs and was ignored, leaving
        # save_best_model always True regardless of what was configured.
        self.save_best_model = save_best
        self.epochs_to_save = []
        self.kwargs = kwargs

    def save_model_parameters(self, trainer, epoch , name = None):
        if name:
            model_parameters = trainer.model.state_dict()
            torch.save(model_parameters, os.path.join(self.dirname, name))
        else:
            model_parameters = trainer.model.state_dict()
            torch.save(model_parameters, os.path.join(self.dirname, f'params_epoch_{epoch}.pth'))

    def on_train_begin(self, trainer, **kwargs):
        max_epochs = trainer.max_epochs
        self.epochs_to_save.append(0)
        # Create a list of epochs to save
        if self.save_each_n_epochs is not None:
            nepochs_to_save = max_epochs // self.save_each_n_epochs
            for i in range(1, nepochs_to_save + 1):
                self.epochs_to_save.append(i * self.save_each_n_epochs)
        # Save all epochs
        self.epochs_to_save.append(max_epochs)

        # Check if dirname exists
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

        self.save_model_parameters(trainer, epoch = 0)

    def on_epoch_end(self, trainer, epoch, **kwargs):
        # Save the model parameters if the current epoch is in the list
        if epoch in self.epochs_to_save:
            self.save_model_parameters(trainer, epoch)

    def on_train_end(self, trainer, **kwargs):
        if self.save_best_model:
            # Save the model parameters of the last epoch
            self.save_model_parameters(trainer, epoch = None, name = 'best_model.pth')
            print(f"Model parameters saved in {self.dirname}")

#%%####################### EARLY STOPPING #####################################################
class EarlyStopping():
    def __init__(self, patience=5, threshold=0.001,
                 monitor='val_loss', lower_is_better=True, load_best=True):
        self.patience = patience
        self.threshold = threshold
        self.monitor = monitor
        self.lower_is_better = lower_is_better
        self.load_best = load_best

        self.best_score = None
        self.epochs_no_improvement = 0
        self.best_model = None

    def on_train_begin(self, trainer, **kwargs):
        self.best_score = None
        self.epochs_no_improvement = 0
        self.best_model = copy.deepcopy(trainer.model)  # Save the initial model

    def on_epoch_end(self, trainer, epoch, **kwargs):
        current_score = trainer.history[self.monitor][-1]
        if self.best_score is None:
            self.best_score = current_score
            self.epochs_no_improvement = 0
            self.best_model = copy.deepcopy(trainer.model)  # Save the best model
        else:
            if self.lower_is_better:
                if current_score < self.best_score - self.threshold:
                    self.best_score = current_score
                    self.epochs_no_improvement = 0
                    self.best_model = copy.deepcopy(trainer.model)
                else:
                    self.epochs_no_improvement += 1
            else:
                if current_score > self.best_score + self.threshold:
                    self.best_score = current_score
                    self.epochs_no_improvement = 0
                    self.best_model = copy.deepcopy(trainer.model)  # Save the best models
                else:
                    self.epochs_no_improvement += 1

        if self.epochs_no_improvement >= self.patience:
            print(f"Early stopping at epoch {epoch}")
            trainer.stop_training = True

    def on_train_end(self, trainer, **kwargs):
        if self.load_best:
            trainer.model = copy.deepcopy(self.best_model)  # Load the best model
            print("Loaded best model parameters.")

################################## LOGGERS #####################################
class LoggerCallback():
    def __init__(self, print_each_n_epochs=1):
        self.print_each_n_epochs = print_each_n_epochs
        self.last_epoch = 0

    def on_train_begin(self, trainer, **kwargs):
        # Initialize the last epoch
        self.last_epoch = 0

    def on_epoch_end(self, trainer, epoch):
        if self.last_epoch == epoch:
            # Obtain the keys of the history dictionary
            keys = trainer.history.keys()
            # Print the keys
            msg = ""
            keys = list(filter(lambda x: x != 'best_epoch', keys))
            for key in keys:
                # Print the key and its value
                msg += f" {key}: {trainer.history[key][-1]:.4f}"
            print(msg)

        elif epoch == self.last_epoch + self.print_each_n_epochs:
            # Obtain the keys of the history dictionary
            keys = trainer.history.keys()
            # Print the keys
            msg = ""
            keys = list(filter(lambda x: x != 'best_epoch', keys))
            for key in keys:
                # Print the key and its value
                msg += f" {key}: {trainer.history[key][-1]:.4f}"
            print(msg)
            # Update the last epoch
            self.last_epoch = epoch
