"""
`IndexSplitStrategy` -- colaborador opcional inyectado en
WithinUnitHoldoutSchema (mismo patrón que LabelTransform: default None
preserva comportamiento actual, un colaborador concreto se activa
explícitamente).

Por qué existe: stratified_sequential_split (slicing secuencial,
determinístico, estratificado por clase) es específico de la semántica
"online simulado" -- simula orden temporal real de llegada de trials.
No tiene sentido para regresión (¿"estratificar por clase" en y
continuo?) ni para cualquier escenario donde no importe simular ese
orden. Extraer el algoritmo de split a un colaborador intercambiable
permite que WithinUnitHoldoutSchema sirva a otras tareas sin cambiar su
propio código -- solo inyectando otra estrategia.

RandomSplit es el default CONCEPTUAL de este módulo (la implementación
más genérica, sin asumir nada sobre orden temporal ni sobre
clasificación) -- pero WithinUnitHoldoutSchema sigue defaulteando
explícitamente a StratifiedSequentialSplit en su propio constructor, para
no cambiar el comportamiento ya verificado y en producción (ver
verify_01_holdout_schema.py). El default de "este módulo en abstracto" y
el default de "este Schema en particular" son decisiones separadas a
propósito -- ver discusión de sesión.
"""
from typing import Any, List, Optional, Protocol, Tuple

import numpy as np


class IndexSplitStrategy(Protocol):
    def split_indices(self, n_total: int, ntest: int, y: Optional[Any] = None) -> Tuple[List[int], List[int]]:
        """Devuelve (train_idx, test_idx). y es opcional -- estrategias
        que no necesitan estratificar (ej. RandomSplit) lo ignoran."""
        ...


def stratified_sequential_split(ntest, labels):
    """Puerto verbatim -- MISMA función que ya vive en schemas.py, sin
    ningún cambio de lógica. Se re-declara acá para que
    StratifiedSequentialSplit pueda envolverla sin importar schemas.py
    (evita import circular, dado que schemas.py va a importar este
    módulo). schemas.py debe seguir teniendo su propia copia idéntica
    hasta que se decida cuál de las dos ubicaciones es la definitiva --
    ver nota de "qué preguntar antes de asumir" al pie."""
    n = len(labels)
    ntrain = n - ntest
    unique_labels = np.unique(labels)
    ntrain_per_class = int(ntrain // len(unique_labels))
    ntest_per_class = int(ntest // len(unique_labels))

    train_idx = []
    test_idx = []
    for cls in unique_labels:
        cls_indices = np.where(labels == cls)[0]
        train_cls_indices = cls_indices[:ntrain_per_class]
        test_cls_indices = cls_indices[ntrain_per_class:ntrain_per_class + ntest_per_class]
        train_idx.extend(train_cls_indices)
        test_idx.extend(test_cls_indices)
    return train_idx, test_idx


class StratifiedSequentialSplit:
    """Envoltorio de stratified_sequential_split, sin cambios internos.
    Específico de clasificación (estratifica por clase) -- para
    regresión, usar RandomSplit."""

    def split_indices(self, n_total, ntest, y=None):
        if y is None:
            raise ValueError("StratifiedSequentialSplit necesita y (labels) para estratificar por clase.")
        return stratified_sequential_split(ntest=ntest, labels=y)


class RandomSplit:
    """Split aleatorio simple, sin estratificar -- default CONCEPTUAL de
    este módulo (ver docstring de arriba). Sirve para cualquier tarea
    (clasificación, regresión) porque no asume nada sobre y."""

    def __init__(self, seed: Optional[int] = None):
        self.seed = seed

    def split_indices(self, n_total, ntest, y=None):
        rng = np.random.RandomState(self.seed)
        idx = rng.permutation(n_total)
        return list(idx[ntest:]), list(idx[:ntest])