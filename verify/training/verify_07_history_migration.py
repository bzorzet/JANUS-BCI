"""
Script de verificación -- migración de `history` de `Trainer_DL` a
`CallbackDispatcher` (Opción 2: property que delega, callbacks.py sin
cambios).

Corré esto a mano (modo debug), no vía pytest. No corras/agregues tests
automatizados -- verificación manual.

Qué verifica:
1. `trainer.history` sigue siendo accesible y da el mismo objeto que
   `trainer.dispatcher.history` -- confirma que es una property que
   delega, no una copia.
2. Un callback real de `callbacks.py` (LossTracker), sin ninguna
   modificación, sigue pudiendo leer/escribir `trainer.history[...]`
   normalmente -- confirma que la migración es transparente.
3. `trainer.get_history()` devuelve el mismo dict que `trainer.history`.
4. Dos Trainer_DL distintos (dos dispatchers distintos) NO comparten
   history -- cada instancia tiene su propio dict, nada quedó compartido
   a nivel de clase por error.
5. Un mini-loop de entrenamiento real (pocas épocas, modelo/datos dummy)
   corre de punta a punta y `trainer.history['train_loss']` queda
   poblado correctamente al final -- prueba de integración, no solo de
   unidad.
"""
import numpy as np
import torch
import torch.nn as nn

from src.training.core.trainer import Trainer_DL
from src.training.core.callback_dispatcher import CallbackDispatcher

# Callback real, sin modificar -- ajustar el import si el path difiere.
from src.torch_utils.callbacks import LossTracker


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def _build_dummy_loader(n=16, batch_size=4):
    X = torch.randn(n, 4)
    y = torch.randint(0, 2, (n,))
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size)


def verify_history_is_delegated_property():
    print("--- 1. trainer.history delega a dispatcher.history (mismo objeto, no copia) ---")
    model = DummyModel()
    trainer = Trainer_DL(
        model=model, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()),
        train_loader=_build_dummy_loader(), max_epochs=1, callbacks=[],
    )

    assert trainer.history is trainer.dispatcher.history, (
        "trainer.history debería ser EXACTAMENTE el mismo objeto que trainer.dispatcher.history "
        "(misma identidad, `is`), no una copia -- si no, escribir en uno no se reflejaría en el otro."
    )
    # Mutar a través de una referencia y confirmar que se ve desde la otra.
    trainer.history["marca_de_prueba"] = 12345
    assert trainer.dispatcher.history["marca_de_prueba"] == 12345, (
        "Escribir en trainer.history no se reflejó en dispatcher.history -- no son el mismo objeto."
    )
    print("  trainer.history is trainer.dispatcher.history -> True")
    print("  mutación cruzada confirmada")
    print("  OK\n")


def verify_existing_callback_unmodified_still_works():
    print("--- 2. LossTracker (callback real, sin modificar) sigue leyendo/escribiendo trainer.history ---")
    model = DummyModel()
    tracker = LossTracker(name="train_loss", on_training=True)
    trainer = Trainer_DL(
        model=model, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()),
        train_loader=_build_dummy_loader(), max_epochs=2, callbacks=[tracker],
    )
    trainer.train()

    assert "train_loss" in trainer.history, (
        "LossTracker debería haber creado la key 'train_loss' en trainer.history "
        "(vía on_train_begin: 'if self.name not in trainer.history: trainer.history[self.name] = []') "
        "-- si no aparece, la migración rompió el acceso del callback."
    )
    assert len(trainer.history["train_loss"]) == 2, (
        f"Se esperaban 2 valores de train_loss (2 épocas), hay {len(trainer.history['train_loss'])}"
    )
    print(f"  trainer.history['train_loss'] = {trainer.history['train_loss']}")
    print("  OK -- LossTracker (sin modificar) funciona igual que antes de la migración\n")


def verify_get_history_matches_property():
    print("--- 3. get_history() devuelve el mismo dict que la property history ---")
    model = DummyModel()
    trainer = Trainer_DL(
        model=model, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()),
        train_loader=_build_dummy_loader(), max_epochs=1, callbacks=[],
    )
    assert trainer.get_history() is trainer.history, (
        "get_history() debería devolver el MISMO objeto que la property history, no una copia."
    )
    print("  trainer.get_history() is trainer.history -> True")
    print("  OK\n")


def verify_two_trainers_dont_share_history():
    print("--- 4. Dos Trainer_DL distintos NO comparten history (nada quedó compartido a nivel de clase) ---")
    model_a = DummyModel()
    model_b = DummyModel()
    trainer_a = Trainer_DL(
        model=model_a, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model_a.parameters()),
        train_loader=_build_dummy_loader(), max_epochs=1, callbacks=[],
    )
    trainer_b = Trainer_DL(
        model=model_b, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model_b.parameters()),
        train_loader=_build_dummy_loader(), max_epochs=1, callbacks=[],
    )

    trainer_a.history["marca_unica_a"] = "solo_en_a"
    assert "marca_unica_a" not in trainer_b.history, (
        "trainer_b.history contiene una key escrita en trainer_a.history -- los dos Trainer están "
        "compartiendo el mismo dict de history (posible bug de mutable default o atributo de clase "
        "en vez de instancia)."
    )
    print("  trainer_a.history y trainer_b.history son dicts independientes")
    print("  OK\n")


def verify_full_training_loop_populates_history():
    print("--- 5. Loop de entrenamiento real (integración) puebla history correctamente ---")
    model = DummyModel()
    tracker_train = LossTracker(name="train_loss", on_training=True)
    val_loader = _build_dummy_loader()
    tracker_val = LossTracker(name="val_loss", on_training=False)

    trainer = Trainer_DL(
        model=model, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()),
        train_loader=_build_dummy_loader(), val_loader=val_loader,
        max_epochs=3, callbacks=[tracker_train, tracker_val],
    )
    trainer.train()

    assert len(trainer.history["train_loss"]) == 3, f"Se esperaban 3 épocas de train_loss, hay {len(trainer.history['train_loss'])}"
    assert len(trainer.history["val_loss"]) == 3, f"Se esperaban 3 épocas de val_loss, hay {len(trainer.history['val_loss'])}"
    assert all(isinstance(v, float) for v in trainer.history["train_loss"]), "train_loss debería contener floats"
    assert trainer.history["best_epoch"] is not None, "best_epoch debería haberse seteado (vía LossTracker.on_train_end)"

    print(f"  train_loss: {[round(v, 4) for v in trainer.history['train_loss']]}")
    print(f"  val_loss:   {[round(v, 4) for v in trainer.history['val_loss']]}")
    print(f"  best_epoch: {trainer.history['best_epoch']}")
    print("  OK -- loop completo de entrenamiento puebla history correctamente end-to-end\n")


if __name__ == "__main__":
    verify_history_is_delegated_property()
    verify_existing_callback_unmodified_still_works()
    verify_get_history_matches_property()
    verify_two_trainers_dont_share_history()
    verify_full_training_loop_populates_history()
    print("=== TODOS LOS CHECKS DE verify_07_history_migration.py PASARON ===")
