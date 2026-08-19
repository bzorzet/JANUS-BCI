"""
`Trainer_DL` refactorizado: compone un `CallbackDispatcher` en vez de
implementar su propio sistema de eventos (repo_viejo/src/torch_utils/trainer.py
mezclaba loop de entrenamiento + gestión de callbacks + políticas de
default). `train()`/`evaluate()`/`infer()`/`model_infer()` se mantienen
funcionalmente idénticos al original.

`self.history` sigue viviendo acá (no se traslada al dispatcher): los
callbacks existentes (`src/torch_utils/callbacks.py`) leen/escriben
`trainer.history` directamente -- moverlo hubiera forzado a reescribir
`callbacks.py`, que el prompt maestro pide migrar tal cual. Decisión
documentada acá en vez de asumida en silencio.

`Trainer_ML` es nuevo (no existía en repo_viejo como clase -- el script
`train_ML_model_within_subject_...` maneja el ciclo fit/predict/predict_proba
inline). Contrato de duck typing, sin heredar de `Trainer_DL` (principio de
composición sobre herencia del refactor).
"""
from typing import Any, Callable, Dict, List, Optional

import torch

from src.training.core.callback_dispatcher import CallbackDispatcher
from src.training.core.loss_strategy import LossStrategy


class Trainer_DL:
    def __init__(
        self,
        model,
        loss_fn,
        optimizer,
        train_loader,
        val_loader=None,
        max_epochs: int = 10,
        callbacks: Optional[List[Any]] = None,
        use_caching: bool = True,
        loss_strategy: Optional[LossStrategy] = None,
    ):
        """
        Args:
            model: modelo de PyTorch a entrenar.
            loss_fn: función de loss default.
            optimizer: optimizer.
            train_loader: DataLoader de entrenamiento.
            val_loader: DataLoader de validación (opcional).
            max_epochs: cantidad de épocas.
            callbacks: lista de instancias de callback YA RESUELTA por el
                orquestador -- este trainer no agrega ningún default (Timer,
                LossTracker, etc.), eso es responsabilidad del caller.
            use_caching: preservado del original (no usado dentro de esta
                clase, igual que en repo_viejo).
            loss_strategy: colaborador Strategy opcional (ver
                src/training/loss_strategy.py). None (default) preserva el
                comportamiento actual: `loss_fn(y_pred, y_true)` directo.

        Nota: el parámetro `test_loader` del `Trainer_DL` original se
        elimina -- estaba muerto (nunca se usaba dentro de la clase); la
        evaluación final sobre test la corre el orquestador vía
        `trainer.infer(test_loader)` después de `train()`.
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.best_model = None
        self.train_loader = train_loader

        self.val_loader = val_loader
        self.val_loader_flag = val_loader is not None

        self.max_epochs = max_epochs
        self.use_caching = use_caching
        self.loss_strategy = loss_strategy

        self.stop_training = False

        self.dispatcher = CallbackDispatcher(callbacks)
        # self.history y initialize_history() ya NO existen acá -- el
        # dispatcher los inicializa en su propio __init__.

    @property
    def history(self) -> Dict[str, Any]:
        """Delega a self.dispatcher.history. Los callbacks existentes
        (callbacks.py) siguen haciendo trainer.history[...] sin saber que
        el dict real vive en el dispatcher -- ver nota en
        callback_dispatcher.py."""
        return self.dispatcher.history

    def _compute_loss(self, y_pred, y_true, X):
        if self.loss_strategy is not None:
            return self.loss_strategy.compute(
                y_pred=y_pred, y_true=y_true, X=X, model=self.model,
                default_loss_fn=self.loss_fn,
            )
        return self.loss_fn(y_pred, y_true)

    def train(self) -> None:
        self.dispatcher.notify('on_train_begin', trainer=self)

        for epoch in range(self.max_epochs):
            self.dispatcher.notify('on_epoch_begin', trainer=self, epoch=epoch)
            self.model.train()
            for batch_idx, (X, y_true) in enumerate(self.train_loader):
                self.dispatcher.notify('on_batch_begin', trainer=self, batch_idx=batch_idx)

                self.optimizer.zero_grad()
                y_pred = self.model_infer(X)
                loss = self._compute_loss(y_pred, y_true, X)

                loss.backward()
                self.optimizer.step()

                self.dispatcher.notify(
                    'on_batch_end', trainer=self, batch_idx=batch_idx,
                    y_pred=y_pred, y_true=y_true, loss=loss.item(), training=True,
                )

            if self.val_loader_flag:
                self.evaluate(epoch)

            self.dispatcher.notify('on_epoch_end', trainer=self, epoch=epoch)

            if self.stop_training:
                break

        self.dispatcher.notify('on_train_end', trainer=self)

    def evaluate(self, epoch) -> None:
        self.model.eval()

        with torch.no_grad():
            for batch_idx, (X, y_true) in enumerate(self.val_loader):
                self.dispatcher.notify('on_batch_begin', trainer=self, batch_idx=batch_idx)
                y_pred = self.model_infer(X)
                loss = self._compute_loss(y_pred, y_true, X)

                self.dispatcher.notify(
                    'on_batch_end', trainer=self, batch_idx=batch_idx,
                    y_pred=y_pred, y_true=y_true, loss=loss.item(), training=False,
                )

    def get_history(self) -> Dict[str, Any]:
        return self.dispatcher.history

    def get_model(self):
        return self.model

    def infer(self, dataloader=None, X=None, probability: bool = True):
        self.model.eval()
        with torch.no_grad():
            if dataloader is not None:
                y_pred = []
                for batch_idx, (X, y_true) in enumerate(dataloader):
                    y_pred.append(self.model_infer(X))
                y_pred = torch.cat(y_pred, dim=0)
            else:
                if X is None:
                    raise ValueError("Either dataloader or X must be provided.")
                y_pred = self.model_infer(X)
        if not probability:
            y_pred = torch.argmax(y_pred, dim=1)
        return y_pred

    def model_infer(self, X):
        if isinstance(X, dict):
            y_pred = self.model(**X)
        else:
            y_pred = self.model(X)
        return y_pred


class Trainer_ML:
    """Contrato duck-typed (sin heredar de Trainer_DL): `train()`,
    `infer(X, probability=True)`, `get_history()`. Confirmado contra
    repo_viejo/train_ML_model_within_subject_online_simulated_for_MI-BCI_classification.py
    (`model.fit(X_train, y_train)`, `model.predict`/`model.predict_proba`,
    pesos persistidos vía `joblib.dump`, no `.pth`).

    X_val/y_val opcionales, sin uso todavía dentro de la clase -- se
    dejan preparados en el constructor para no romper la interfaz cuando
    se implemente selección de mejor modelo/hiperparámetros vía val
    (planeado a futuro, coherente con que Trainer_DL también use val con
    ese propósito, para poder comparar ML y DL bajo el mismo criterio de
    selección). Hoy, val simplemente no se usa: get_history() sigue
    devolviendo {} y train() sigue sin tocarlos. Cualquier caller
    existente que instancie Trainer_ML(model, X_train, y_train) sin val
    sigue funcionando idéntico -- default None, sin cambio de
    comportamiento."""

    def __init__(self, model, X_train, y_train, X_val=None, y_val=None):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def train(self) -> None:
        self.model.fit(self.X_train, self.y_train)

    def infer(self, X, probability: bool = True):
        if probability:
            return self.model.predict_proba(X)
        return self.model.predict(X)

    def get_history(self) -> Dict[str, Any]:
        return {}