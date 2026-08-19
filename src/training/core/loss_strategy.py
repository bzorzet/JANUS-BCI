"""
Contrato Strategy para cómputo de loss custom, inyectado en `Trainer_DL`
por separado del `CallbackDispatcher` (no es un evento de notify -- una
loss tiene que devolver un tensor listo para `.backward()`, un callback no
tiene contrato de retorno).

Sin implementaciones concretas en este refactor -- solo el contrato y el
punto de inyección en `Trainer_DL`. `loss_strategy=None` (default)
preserva el comportamiento actual (`loss_fn(y_pred, y_true)` directo) para
todo config existente.
"""


class LossStrategy:
    def compute(self, y_pred, y_true, X, model, default_loss_fn):
        """Debe devolver un tensor de loss listo para .backward()."""
        raise NotImplementedError
