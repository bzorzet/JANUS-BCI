"""
`TrainingSchema` -- reemplaza el diseño anterior de Splitter + PartitionSpec
+ materialize_fold. Cada Schema unifica en una sola unidad lo que antes
eran dos responsabilidades separadas (decidir la partición + materializar
datos reales), recibiendo un `DataProvider` en su constructor en vez de
exponer un paso intermedio serializable hacia el caller.

Vocabulario: "unit" reemplaza a "subject" en TODO este módulo -- es el
único cambio de nombre que hace que este archivo sea reusable fuera de
BCI (ver src/training/data_provider.py para dónde vive el vocabulario
específico de EEG).

Cardinalidad uniforme entre los 3 Schemas: una llamada a generate_folds()
por "unidad de foco" (unit_id para los within-unit, test_unit para LOSO)
-- el caller (orquestador) itera, ningún Schema genera el barrido completo
de una sola llamada.

Contrato común (duck-typed, sin herencia real):
    class TrainingSchema(Protocol):
        def generate_folds(self, **context) -> List[Fold]: ...
        def partition_name(self, fold: Fold) -> str: ...

`label_transform` (LabelTransform, ver execution/label_transform.py) es un
colaborador opcional inyectado en cada Schema -- default
ClassificationLabelTransform() preserva el comportamiento actual
(codificación de clases) para configs existentes. Ningún Schema llama a
encode_labels directamente: eso mantendría el Schema atado a clasificación,
que es justo lo que este colaborador evita -- ver docstring de
label_transform.py.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.training.data_provider import DataProvider
from src.training.core.label_transform import ClassificationLabelTransform, LabelTransform
from src.training.splitting.fold import Fold
from src.training.splitting.index_split_strategy import IndexSplitStrategy, StratifiedSequentialSplit

class WithinUnitHoldoutSchema:
    def __init__(
        self, provider: DataProvider, ptrain: float, pval: float, ptest: float,
        split_strategy: Optional[IndexSplitStrategy] = None,
        label_transform: Optional[LabelTransform] = None,
    ):
        self.provider = provider
        self.ptrain = ptrain
        self.pval = pval
        self.ptest = ptest
        # Default explícito de ESTE Schema, no el default conceptual del
        # módulo -- ver docstring de index_split_strategy.py.
        self.split_strategy = split_strategy or StratifiedSequentialSplit()
        self.label_transform = label_transform or ClassificationLabelTransform()

    def generate_folds(self, **context) -> List[Fold]:
        unit_id = context["unit_id"]
        X, y, metadata = self.provider.get_unit(unit_id)
        n = len(y)

        train_idx_intermediate, test_idx = self.split_strategy.split_indices(n_total=n, ntest=int(self.ptest * n), y=y)
        y_train_intermediate = y[train_idx_intermediate]

        train_idx_relative, val_idx_relative = self.split_strategy.split_indices(
            n_total=len(train_idx_intermediate), ntest=int(self.pval * n), y=y_train_intermediate
        )
        train_idx = [train_idx_intermediate[i] for i in train_idx_relative]
        val_idx = [train_idx_intermediate[i] for i in val_idx_relative]

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        meta_train = metadata.iloc[train_idx] if metadata is not None else None
        meta_val = metadata.iloc[val_idx] if metadata is not None else None
        meta_test = metadata.iloc[test_idx] if metadata is not None else None

        y_train_t, y_val_t, y_test_t, label_meta = self.label_transform.transform(y_train, y_val, y_test)

        return [Fold(
            X_train=X_train, y_train=y_train_t,
            X_val=X_val, y_val=y_val_t,
            X_test=X_test, y_test=y_test_t,
            metadata={
                "unit_id": unit_id,
                "train_idx": [int(i) for i in train_idx],
                "val_idx": [int(i) for i in val_idx],
                "test_idx": [int(i) for i in test_idx],
                **label_meta,
                "metadata_train": meta_train, "metadata_val": meta_val, "metadata_test": meta_test,
            },
        )]

    def partition_name(self, fold: Fold) -> str:
        return f"unit_{fold.metadata['unit_id']:02d}"


class WithinUnitKFoldSchema:
    """K-fold aleatorio, estratificado por clase, dentro de una misma
    unidad -- para estimación robusta de performance vía cross-validation.
    Algoritmo DISTINTO de WithinUnitHoldoutSchema (que es secuencial y
    determinístico): acá la pregunta experimental es otra, y el split es
    aleatorio. NO reutiliza stratified_sequential_split. El test se separa
    UNA sola vez (aleatorio estratificado, congelado) y es compartido por
    los K folds -- no se re-separa test en cada fold."""

    def __init__(
        self, provider: DataProvider, k_folds: int, ptest: float, seed: int,
        label_transform: Optional[LabelTransform] = None,
    ):
        self.provider = provider
        self.k_folds = k_folds
        self.ptest = ptest
        self.seed = seed
        self.label_transform = label_transform or ClassificationLabelTransform()

    def generate_folds(self, **context) -> List[Fold]:
        """context esperado: unit_id."""
        unit_id = context["unit_id"]
        X, y, metadata = self.provider.get_unit(unit_id)

        all_idx = np.arange(len(y))
        test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.ptest, random_state=self.seed)
        train_val_idx, test_idx = next(test_splitter.split(all_idx, y))

        y_train_val = y[train_val_idx]
        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)

        X_test, y_test = X[test_idx], y[test_idx]
        meta_test = metadata.iloc[test_idx] if metadata is not None else None

        folds = []
        for fold_idx, (train_relative, val_relative) in enumerate(kfold.split(train_val_idx, y_train_val)):
            train_idx = train_val_idx[train_relative]
            val_idx = train_val_idx[val_relative]

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]
            meta_train = metadata.iloc[train_idx] if metadata is not None else None
            meta_val = metadata.iloc[val_idx] if metadata is not None else None

            y_train_t, y_val_t, y_test_t, label_meta = self.label_transform.transform(y_train, y_val, y_test)

            folds.append(Fold(
                X_train=X_train, y_train=y_train_t,
                X_val=X_val, y_val=y_val_t,
                X_test=X_test, y_test=y_test_t,
                metadata={
                    "unit_id": unit_id, "fold_idx": fold_idx, "k_folds": self.k_folds,
                    "train_idx": [int(i) for i in train_idx],
                    "val_idx": [int(i) for i in val_idx],
                    "test_idx": [int(i) for i in test_idx],
                    **label_meta,
                    "metadata_train": meta_train, "metadata_val": meta_val, "metadata_test": meta_test,
                },
            ))
        return folds

    def partition_name(self, fold: Fold) -> str:
        # Nota: con K folds, varios Fold devueltos por la misma llamada
        # comparten partition_name -- sub-organización por fold_idx queda
        # a cargo del orquestador/guardado (fuera de alcance acá).
        return f"unit_{fold.metadata['unit_id']:02d}"


class LeaveOneUnitOutSchema:
    """Particiona a nivel de UNIDAD completa -- nunca mezcla trials de una
    misma unidad entre train y val. Una llamada a generate_folds = un
    fold (la unidad indicada en test_unit es el test; val_units es una
    muestra aleatoria de las unidades restantes). No toca directamente
    ningún dataset -- todo pasa por self.provider."""

    def __init__(
        self, provider: DataProvider, pval_subjects: float, seed: int,
        label_transform: Optional[LabelTransform] = None,
    ):
        self.provider = provider
        self.pval_subjects = pval_subjects
        self.seed = seed
        self.label_transform = label_transform or ClassificationLabelTransform()

    def generate_folds(self, **context) -> List[Fold]:
        """context esperado: test_unit (UN identificador, no una lista --
        misma cardinalidad que los otros 2 Schemas)."""
        test_unit = context["test_unit"]
        remaining = [u for u in self.provider.list_units() if u != test_unit]

        rng = np.random.RandomState(self.seed)
        n_val = round(self.pval_subjects * len(remaining))
        # REGLA CRÍTICA, no negociable: ningún unit simultáneamente en
        # train_units y val_units -- la exclusión de abajo lo garantiza.
        if n_val > 0:
            val_units = [int(u) for u in rng.choice(remaining, size=n_val, replace=False)]
        else:
            val_units = []
        train_units = [u for u in remaining if u not in val_units]

        X_train, y_train, meta_train = self.provider.get_units(train_units)
        if val_units:
            X_val, y_val, meta_val = self.provider.get_units(val_units)
        else:
            X_val, y_val, meta_val = None, None, None
        X_test, y_test, meta_test = self.provider.get_unit(test_unit)

        y_train_t, y_val_t, y_test_t, label_meta = self.label_transform.transform(y_train, y_val, y_test)

        return [Fold(
            X_train=X_train, y_train=y_train_t,
            X_val=X_val, y_val=y_val_t,
            X_test=X_test, y_test=y_test_t,
            metadata={
                "test_unit": test_unit,
                "train_units": train_units,
                "val_units": val_units,
                **label_meta,
                "metadata_train": meta_train, "metadata_val": meta_val, "metadata_test": meta_test,
            },
        )]

    def partition_name(self, fold: Fold) -> str:
        return f"loo_test-unit_{fold.metadata['test_unit']:02d}"


SCHEMA_REGISTRY = {
    "WithinUnitHoldoutSchema": WithinUnitHoldoutSchema,
    "WithinUnitKFoldSchema": WithinUnitKFoldSchema,
    "LeaveOneUnitOutSchema": LeaveOneUnitOutSchema,
}