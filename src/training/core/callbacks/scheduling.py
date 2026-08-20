

class SchedulerStepCallback:
    """Para schedulers que NO dependen de una métrica (StepLR,
    CosineAnnealingLR, warmup schedulers) -- .step() sin argumentos, en
    el momento que corresponda (por época o por batch)."""

    def __init__(self, scheduler, step_on: str = "epoch"):
        self.scheduler = scheduler
        self.step_on = step_on

    def on_epoch_end(self, trainer, epoch, **kwargs):
        if self.step_on == "epoch":
            self.scheduler.step()

    def on_batch_end(self, trainer, batch_idx, training=True, **kwargs):
        if self.step_on == "batch" and training:
            self.scheduler.step()


class SchedulerPlateauCallback:
    """Para ReduceLROnPlateau -- necesita la métrica en cada .step(), se
    lee de trainer.history. IMPORTANTE: debe ir DESPUÉS de LossTracker en
    la lista de callbacks del config (LossTracker escribe el valor de
    esta época en trainer.history antes de que este callback lo lea, en
    el mismo hook on_epoch_end)."""

    def __init__(self, scheduler, monitor: str = "val_loss"):
        self.scheduler = scheduler
        self.monitor = monitor

    def on_epoch_end(self, trainer, epoch, **kwargs):
        if self.monitor in trainer.history and trainer.history[self.monitor]:
            current_value = trainer.history[self.monitor][-1]
            self.scheduler.step(current_value)