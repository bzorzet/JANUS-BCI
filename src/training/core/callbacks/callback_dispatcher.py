"""
Sistema de eventos extraído de `Trainer_DL` (antes `initialize_callbacks`/
`notify` en repo_viejo/src/torch_utils/trainer.py). Composición, no
herencia -- `Trainer_DL` compone un `CallbackDispatcher`, no lo implementa.

Diferencia clave respecto al original: los callbacks "default"
(`TimerCallback`, `LossTracker` x2) ya NO se agregan automáticamente acá --
es responsabilidad del orquestador (`src/training/orchestrator.py`) armar la
lista completa (incluyendo los que antes eran default) y pasarla ya
resuelta.

`history` migró acá desde `Trainer_DL` (antes vivía como atributo directo
del Trainer): es estado que los CALLBACKS llenan (LossTracker, EpochScoring,
etc.), no algo que el Trainer calcule -- alojar ese estado compartido es
más responsabilidad del dispatcher de eventos que del ejecutor del loop de
entrenamiento (ver discusión de SRP). `Trainer_DL.history` sigue existiendo
como property que delega acá, así que `callbacks.py` NO se modifica --
cada callback sigue haciendo `trainer.history[...]` exactamente igual que
antes, sin saber que el dict ahora vive en el dispatcher.
"""
from typing import Any, Dict, List, Optional


class CallbackDispatcher:
    EVENTS = [
        'on_train_begin', 'on_train_end', 'on_epoch_begin',
        'on_epoch_end', 'on_batch_begin', 'on_batch_end',
    ]

    def __init__(self, callbacks: Optional[List[Any]] = None):
        self.callbacks: Dict[str, List[Any]] = {event: [] for event in self.EVENTS}
        for cb in (callbacks or []):
            for event in self.EVENTS:
                if hasattr(cb, event):
                    self.callbacks[event].append(cb)

        # Migrado desde Trainer_DL.initialize_history() -- mismo contenido
        # inicial, ahora alojado acá.
        self.history: Dict[str, Any] = {}
        self.history['epoch'] = []
        self.history['best_epoch'] = None

    def notify(self, event: str, trainer, **kwargs) -> None:
        """`trainer` se pasa posicional -- igual que el `notify` original,
        que llamaba `getattr(cb, method_name)(self, **cb_kwargs)`. Los
        callbacks existentes (callbacks.py) leen/escriben
        `trainer.history`/`trainer.model`/`trainer.stop_training`
        directamente, así que necesitan esta referencia sin cambios de
        firma. `trainer.history` es ahora una property que delega a
        `self.history` (este dict) -- transparente para los callbacks."""
        for cb in self.callbacks[event]:
            getattr(cb, event)(trainer, **kwargs)