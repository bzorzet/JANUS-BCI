import time
#%%################################################# SIMPLE CALLBACKS ####################################################
class TimerCallback():
    def on_train_begin(self, trainer):
        self.start_time = time.time()

    def on_train_end(self, trainer):
        elapsed_time = time.time() - self.start_time
        print(f"Training completed in {elapsed_time:.2f} seconds")
        trainer.history['duration'] = elapsed_time
