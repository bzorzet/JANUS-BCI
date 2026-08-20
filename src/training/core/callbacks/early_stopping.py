import copy 
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
