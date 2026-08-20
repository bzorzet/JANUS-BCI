
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

