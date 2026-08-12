"""
Contrato de particionamiento del lado training (DESIGN, prompt maestro
sección 5). `stratified_sequential_split` es puerto verbatim de
`repo_viejo/src/cross_validators/splitter.py` -- determinístico (slicing
secuencial por clase, floor-division), sin aleatoriedad.

Decisión ya cerrada con el usuario: el config real
(`general_script_config`) no tiene `k_folds` -- tiene `ptrain/pval/ptest`
(split único) más `model_init_seed` como lista de repeticiones de init de
pesos, ajeno al split. En el código viejo el argumento seed nunca afecta el
split en sí. `WithinSubjectHoldoutSplitter.split()` preserva esto exacto:
produce UN partition determinístico por sujeto, sin parámetro de seed. El
loop de repeticiones por `model_init_seed` es responsabilidad del
orquestador (`src/training/orchestrator.py`), no de este módulo.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np


def stratified_sequential_split(ntest, labels):
    """Puerto verbatim de repo_viejo/src/cross_validators/splitter.py.
    Determinístico: para cada clase, toma las primeras `ntrain_per_class`
    muestras (en el orden en que aparecen en `labels`) como train y las
    siguientes `ntest_per_class` como test -- floor-division por clase, así
    que si `n` no es divisible exactamente algunas muestras al final de cada
    clase quedan fuera de ambos conjuntos (comportamiento preservado tal
    cual, no es un bug a corregir acá)."""
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


@dataclass
class Fold:
    X_train: Any
    y_train: Any
    X_val: Any
    y_val: Any
    X_test: Any
    y_test: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    # metadata esperado: subject_id, train_idx, val_idx, test_idx (int lists,
    # serializables), metadata_train/val/test (DataFrames de contexto por
    # trial) -- insumo tanto para el equivalente de info_data_training.json
    # como para filas de run_factors.


class WithinSubjectHoldoutSplitter:
    """Split único, determinístico, estratificado por clase, dentro de un
    mismo sujeto. Nombre elegido explícitamente en vez de "KFold": no hay
    folds ni repeticiones acá -- ver docstring del módulo."""

    def __init__(self, ptrain: float, pval: float, ptest: float):
        self.ptrain = ptrain
        self.pval = pval
        self.ptest = ptest

    def split(self, X, y, metadata=None, subject_id: Optional[Any] = None) -> Fold:
        n = len(y)

        # ntest como fracción del largo ORIGINAL de y -- preservado tal cual
        # el script viejo (no relativo al remanente de train).
        train_idx, test_idx = stratified_sequential_split(ntest=self.ptest * n, labels=y)
        X_test, y_test = X[test_idx], y[test_idx]
        X_train, y_train = X[train_idx], y[train_idx]
        metadata_test = metadata.iloc[test_idx] if metadata is not None else None
        metadata_train = metadata.iloc[train_idx] if metadata is not None else None

        # pval también como fracción del largo ORIGINAL de y (no de
        # len(y_train)). `train_idx` se reasigna acá (shadowing intencional,
        # igual que repo_viejo): el `train_idx` final guardado en metadata
        # queda relativo al X_train de este segundo split, no al X original
        # -- comportamiento preservado verbatim, no es una inconsistencia a
        # corregir en este refactor (no forma parte de los 3 fixes ya
        # aprobados).
        train_idx, val_idx = stratified_sequential_split(ntest=self.pval * n, labels=y_train)
        X_val, y_val = X_train[val_idx], y_train[val_idx]
        metadata_val = metadata_train.iloc[val_idx] if metadata_train is not None else None
        X_train, y_train = X_train[train_idx], y_train[train_idx]
        metadata_train = metadata_train.iloc[train_idx] if metadata_train is not None else None

        return Fold(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val,
            X_test=X_test, y_test=y_test,
            metadata={
                "subject_id": subject_id,
                "train_idx": [int(i) for i in train_idx],
                "val_idx": [int(i) for i in val_idx],
                "test_idx": [int(i) for i in test_idx],
                "metadata_train": metadata_train,
                "metadata_val": metadata_val,
                "metadata_test": metadata_test,
            },
        )


SPLITTER_REGISTRY = {
    "WithinSubjectHoldoutSplitter": WithinSubjectHoldoutSplitter,
}
