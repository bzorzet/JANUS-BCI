import numpy as np
import torch

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

