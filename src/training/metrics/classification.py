"""
`ClassificationMetrics` -- migra _compute_test_metrics de
orchestrator_testing.py, generalizada a N clases y con el set de métricas
a calcular CONFIGURABLE (antes era fijo: siempre accuracy+auc+recall+
precision). Default = solo "accuracy" -- CAMBIO DE COMPORTAMIENTO
consciente respecto al código anterior (que siempre calculaba las 4);
cualquier config que necesite el resto debe especificarlo explícito en
`metrics`.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)


def _compute_accuracy(y_true, y_pred, y_pred_proba, classes, split, **kwargs):
    return [{"split": split, "metric_name": "accuracy", "value": accuracy_score(y_true, y_pred)}]


def _compute_auc(y_true, y_pred, y_pred_proba, classes, split, auc_multiclass="ovr", **kwargs):
    n_classes = len(classes)
    auc = np.nan
    if y_pred_proba is not None and n_classes > 1:
        if n_classes == 2:
            auc = roc_auc_score(y_true, y_pred_proba[:, 1])
        else:
            # N>2 clases: necesita TODAS las probabilidades, no una sola
            # columna -- esto es lo que estaba roto en el código anterior.
            auc = roc_auc_score(y_true, y_pred_proba, multi_class=auc_multiclass)
    return [{"split": split, "metric_name": "auc", "value": auc}]


def _compute_recall(y_true, y_pred, y_pred_proba, classes, split, **kwargs):
    # average=None + labels=classes: un valor por clase, generaliza el
    # "for cls in (0, 1)" fijo que había antes (pos_label solo sirve en
    # binario, no escala a N clases).
    values = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return [{"split": split, "metric_name": f"recall_{cls}", "value": v} for cls, v in zip(classes, values)]


def _compute_precision(y_true, y_pred, y_pred_proba, classes, split, **kwargs):
    values = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return [{"split": split, "metric_name": f"precision_{cls}", "value": v} for cls, v in zip(classes, values)]


def _compute_f1(y_true, y_pred, y_pred_proba, classes, split, **kwargs):
    values = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    return [{"split": split, "metric_name": f"f1_{cls}", "value": v} for cls, v in zip(classes, values)]


METRIC_FUNCTIONS = {
    "accuracy": _compute_accuracy,
    "auc": _compute_auc,
    "recall": _compute_recall,
    "precision": _compute_precision,
    "f1": _compute_f1,
}


class ClassificationMetrics:
    """`metrics`: lista de nombres a calcular, de METRIC_FUNCTIONS.
    Default = ["accuracy"] únicamente -- cambio de comportamiento
    consciente respecto al código anterior, confirmado con el usuario.

    `auc_multiclass`: "ovr" (default) u "ovo", pasado a roc_auc_score
    cuando hay más de 2 clases y "auc" está en `metrics`.

    `recall`/`precision`/`f1` devuelven UNA FILA POR CLASE (ej.
    recall_0, recall_1, ..., recall_N) -- no un escalar agregado."""

    def __init__(self, metrics: Optional[List[str]] = None, auc_multiclass: str = "ovr"):
        self.metrics = metrics or ["accuracy"]
        self.auc_multiclass = auc_multiclass

        unknown = [m for m in self.metrics if m not in METRIC_FUNCTIONS]
        if unknown:
            raise ValueError(
                f"Métricas desconocidas en 'metrics': {unknown}. "
                f"Disponibles: {list(METRIC_FUNCTIONS.keys())}"
            )

    def compute(self, y_true, y_pred, y_pred_proba=None, split: str = "test") -> List[Dict[str, Any]]:
        classes = np.unique(y_true)
        rows: List[Dict[str, Any]] = []
        for metric_name in self.metrics:
            fn = METRIC_FUNCTIONS[metric_name]
            rows.extend(fn(
                y_true=y_true, y_pred=y_pred, y_pred_proba=y_pred_proba,
                classes=classes, split=split, auc_multiclass=self.auc_multiclass,
            ))
        return rows