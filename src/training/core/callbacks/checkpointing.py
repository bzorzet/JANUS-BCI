import numpy as np
import torch
import time
import os
import copy


#%% ################################################# CHECKPOINTS ####################################################
class CustomCheckpoint():
    def __init__(self, dirname, save_each_n_epochs=None, save_best=True, prefix="", **kwargs):
        self.dirname = dirname
        self.save_each_n_epochs = save_each_n_epochs
        self.save_best_model = save_best
        # NUEVO: prefijo antepuesto a cada nombre de archivo -- permite
        # que varias seeds compartan la misma carpeta (guardado aplanado,
        # Paso 4b) sin pisarse entre sí. Default "" preserva el
        # comportamiento anterior si alguien lo instancia sin prefix.
        self.prefix = prefix
        self.epochs_to_save = []
        self.kwargs = kwargs

    def save_model_parameters(self, trainer, epoch, name=None):
        if name:
            filename = f"{self.prefix}{name}" if self.prefix else name
        else:
            filename = f"{self.prefix}epoch_{epoch}.pth" if self.prefix else f"params_epoch_{epoch}.pth"
        model_parameters = trainer.model.state_dict()
        torch.save(model_parameters, os.path.join(self.dirname, filename))

    def on_train_end(self, trainer, **kwargs):
        if self.save_best_model:
            self.save_model_parameters(trainer, epoch=None, name="best.pth")
            print(f"Model parameters saved in {self.dirname}")

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
