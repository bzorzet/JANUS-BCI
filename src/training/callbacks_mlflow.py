"""
Callback de MLflow -- solo loguea DENTRO de un run ya abierto por el
orquestador (Camino B, prompt maestro sección 6.1). No conoce el config del
proyecto, no arma run_name, no decide tags, no abre ni cierra runs: eso es
responsabilidad exclusiva del orquestador (`mlflow.start_run(...)` como
context manager alrededor de `Trainer_DL.train()`).
"""
import mlflow


class MLflowCallback:
    def on_epoch_end(self, trainer, epoch, **kwargs):
        for key, values in trainer.history.items():
            if key in ("epoch", "best_epoch"):
                continue
            if not values:
                continue
            value = values[-1] if isinstance(values, list) else values
            if not isinstance(value, (int, float)):
                continue
            mlflow.log_metric(key, value, step=epoch)
