"""
`Tester_DL`/`Tester_ML` -- contrato simple, sin sistema de eventos (no es un
loop de épocas, es un solo forward pass). El logging (CSV, MLflow) es
responsabilidad del orquestador que los usa, no de estas clases.

`Tester_DL` es puerto de `Tester` en
`repo_viejo/src/torch_utils/trainer.py`, con un fix puntual ya aprobado:
`model_infer(X)` se llamaba DOS VECES por batch en el original (cómputo
redundante, no cambiaba resultados numéricos) -- acá se llama una sola vez.
"""
from typing import Any, Dict, Optional

import torch


class Tester_DL:
    def __init__(self, model, loss_fn=None):
        self.model = model
        self.loss_fn = loss_fn

    def test(self, test_loader, probability: bool = True):
        """Devuelve (y_pred, loss). Sin CallbackDispatcher, sin eventos por
        batch -- explícitamente fuera de esta clase (prompt maestro sección
        4)."""
        self.model.eval()
        with torch.no_grad():
            y_pred_list = []
            losses = []
            for batch_idx, (X, y_true) in enumerate(test_loader):
                y_pred = self.model_infer(X)  # llamado una sola vez (fix del bug original)
                y_pred_list.append(y_pred)
                if self.loss_fn is not None:
                    losses.append(self.loss_fn(y_pred, y_true))
            y_pred = torch.cat(y_pred_list, dim=0)

            loss = None
            if self.loss_fn is not None:
                loss = torch.mean(torch.stack(losses))
        if not probability:
            y_pred = torch.argmax(y_pred, dim=1)
        return y_pred, loss

    def model_infer(self, X):
        if isinstance(X, dict):
            y_pred = self.model(**X)
        else:
            y_pred = self.model(X)
        return y_pred


class Tester_ML:
    """Contrato análogo a Tester_DL para modelos sklearn-like. Sin loss (los
    modelos ML clásicos evaluados acá no tienen un concepto de loss --
    mismo comportamiento que el script de testing ML viejo, que solo
    reporta accuracy/AUC/recall/precision, no loss)."""

    def __init__(self, model):
        self.model = model

    def test(self, X, probability: bool = True):
        if probability:
            y_pred = self.model.predict_proba(X)
        else:
            y_pred = self.model.predict(X)
        return y_pred, None
