

class Callback:
    def on_train_begin(self, trainer): pass
    def on_epoch_begin(self, trainer, epoch): pass
    def on_batch_begin(self, trainer, batch_idx): pass
    def on_batch_end(self, trainer, batch_idx): pass
    def on_epoch_end(self, trainer, epoch): pass
    def on_train_end(self, trainer): pass