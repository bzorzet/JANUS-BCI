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

Contrato (sesión de diseño nueva -- distingue "decidir la partición" de
"obtener los datos"): un `Splitter` nunca recibe el objeto `dataset` ni
toca `X`/`y` reales de más de un sujeto a la vez -- solo trabaja con
índices/IDs y devuelve una lista de `PartitionSpec` (descripción de la
partición, sin datos). `materialize_fold` es la única función que traduce
un `PartitionSpec` a un `Fold` con datos reales, vía
`dataset.flatten_subject_data`/`flatten_pool_data`. La estratificación de
los dos splitters within-subject no depende de que `y` esté binarizado
(`np.unique` funciona con cualquier encoding de 2 clases) -- por eso
`encode_labels` se aplica en `materialize_fold`, como último paso antes de
devolver el `Fold`, y no antes de llamar a `Splitter.split()`.
"""
from typing import Any, List, Optional

import numpy as np
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from src.training.splitting.fold import PartitionSpec

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



class WithinSubjectHoldoutSplitter:
    """Split único, determinístico, estratificado por clase, dentro de un
    mismo sujeto. Nombre elegido explícitamente en vez de "KFold": no hay
    folds ni repeticiones acá -- ver docstring del módulo."""

    def __init__(self, ptrain: float, pval: float, ptest: float):
        self.ptrain = ptrain
        self.pval = pval
        self.ptest = ptest

    def split(self, X, y, metadata=None, subject_id: Optional[Any] = None) -> List[PartitionSpec]:
        n = len(y)

        # ntest como fracción del largo ORIGINAL de y -- preservado tal cual
        # el script viejo (no relativo al remanente de train).
        train_idx_intermediate, test_idx = stratified_sequential_split(ntest=self.ptest * n, labels=y)
        y_train_intermediate = y[train_idx_intermediate]

        # pval también como fracción del largo ORIGINAL de y (no de
        # len(y_train_intermediate)). Los índices que devuelve este segundo
        # split quedan relativos a y_train_intermediate -- se traducen de
        # vuelta al espacio de índices del X/y original ANTES de perder
        # train_idx_intermediate (fix ya aprobado con el usuario: antes
        # quedaban relativos al resultado intermedio, ahora quedan en el
        # mismo espacio de referencia que test_idx). El algoritmo en sí
        # (qué trials terminan en cada split) no cambia.
        train_idx_relative, val_idx_relative = stratified_sequential_split(
            ntest=self.pval * n, labels=y_train_intermediate
        )
        train_idx = [train_idx_intermediate[i] for i in train_idx_relative]
        val_idx = [train_idx_intermediate[i] for i in val_idx_relative]

        return [
            PartitionSpec(
                train_idx=[int(i) for i in train_idx],
                val_idx=[int(i) for i in val_idx],
                test_idx=[int(i) for i in test_idx],
                metadata={"subject_id": subject_id},
            )
        ]


class WithinSubjectKFold:
    """K-fold aleatorio, estratificado por clase, dentro de un mismo
    sujeto -- para estimación robusta de performance vía cross-validation.
    Algoritmo DISTINTO de WithinSubjectHoldoutSplitter (que es secuencial y
    determinístico, simula orden temporal real de llegada de trials): acá
    la pregunta experimental es otra, y el split es aleatorio. No reutiliza
    `stratified_sequential_split`. El test se separa UNA sola vez (aleatorio
    estratificado, congelado) y es compartido por los K folds -- no se
    re-separa test en cada fold."""

    def __init__(self, k_folds: int, ptest: float, seed: int):
        self.k_folds = k_folds
        self.ptest = ptest
        self.seed = seed

    def split(self, X, y, metadata=None, subject_id: Optional[Any] = None) -> List[PartitionSpec]:
        

        all_idx = np.arange(len(y))

        test_splitter = StratifiedShuffleSplit(n_splits=1, test_size=self.ptest, random_state=self.seed)
        train_val_idx, test_idx = next(test_splitter.split(all_idx, y))

        y_train_val = y[train_val_idx]

        kfold = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.seed)

        specs = []
        for fold_idx, (train_relative, val_relative) in enumerate(kfold.split(train_val_idx, y_train_val)):
            train_idx = train_val_idx[train_relative]
            val_idx = train_val_idx[val_relative]
            specs.append(
                PartitionSpec(
                    train_idx=[int(i) for i in train_idx],
                    val_idx=[int(i) for i in val_idx],
                    test_idx=[int(i) for i in test_idx],
                    metadata={"subject_id": subject_id, "fold_idx": fold_idx, "k_folds": self.k_folds},
                )
            )
        return specs


class LeaveOneSubjectOut:
    """Particiona a nivel de SUJETO completo -- nunca mezcla trials de un
    mismo sujeto entre train y val (evita que el modelo memorice patrones
    idiosincráticos de un sujeto compartido entre ambos splits, lo cual
    invalidaría val como proxy de generalización a sujetos nuevos). Un fold
    por sujeto en subject_list (ese sujeto = test); val_subjects es una
    muestra aleatoria de los sujetos restantes. No toca el dataset ni carga
    nada -- solo decide qué IDs van a cada split (ver docstring del
    módulo)."""

    def __init__(self, pval_subjects: float, seed: int):
        self.pval_subjects = pval_subjects
        self.seed = seed

    def split(self, subject_list: List[int], metadata=None) -> List[PartitionSpec]:
        rng = np.random.RandomState(self.seed)

        specs = []
        for test_subject in subject_list:
            remaining = [s for s in subject_list if s != test_subject]
            n_val = round(self.pval_subjects * len(remaining))
            if n_val > 0:
                val_subjects = [int(s) for s in rng.choice(remaining, size=n_val, replace=False)]
            else:
                val_subjects = []
            train_subjects = [s for s in remaining if s not in val_subjects]

            specs.append(
                PartitionSpec(
                    train_subjects=train_subjects,
                    val_subjects=val_subjects,
                    test_subjects=[test_subject],
                    metadata={
                        "test_subject": test_subject,
                        "train_subjects": train_subjects,
                        "val_subjects": val_subjects,
                    },
                )
            )
        return specs


SPLITTER_REGISTRY = {
    "WithinSubjectHoldoutSplitter": WithinSubjectHoldoutSplitter,
    "WithinSubjectKFold": WithinSubjectKFold,
    "LeaveOneSubjectOut": LeaveOneSubjectOut,
}
