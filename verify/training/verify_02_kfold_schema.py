"""
Script de verificación 2/N (REESCRITO) -- WithinUnitKFoldSchema.

Corré esto a mano (modo debug), no vía pytest.

Cambio más importante respecto a la versión anterior: acá es donde el
check de "el provider se llama 1 sola vez" importa más -- con K folds de
la MISMA unidad, si generate_folds() llamara a provider.get_unit() dentro
del loop de folds en vez de una vez antes del loop, se leerían los datos
K veces de más. Verificado explícitamente (check 6).

Qué verifica (mismos invariantes que antes + el nuevo):
1. Test congelado, compartido por los K folds.
2. Sin overlap y cobertura completa dentro de cada fold.
3. Folds distintos tienen train/val distintos entre sí.
4. Reproducibilidad: mismo seed -> mismo resultado exacto.
5. Los índices apuntan al X original (valores == índice+1).
6. NUEVO: provider.get_unit() se llama exactamente 1 vez, sin importar
   cuántos folds (k_folds) se generen.
7. NUEVO: cada trial del pool (train+val) es val en EXACTAMENTE 1 fold
   (invariante agregado en sesión anterior, no estaba en la v1 original).
"""
import numpy as np
import pandas as pd

from src.training.splitting.schemas import WithinUnitKFoldSchema


class FakeDataProvider:
    def __init__(self, X, y, metadata=None):
        self._X, self._y, self._metadata = X, y, metadata
        self.get_unit_calls = 0

    def get_unit(self, unit_id, **kwargs):
        self.get_unit_calls += 1
        return self._X, self._y, self._metadata

    def get_units(self, unit_ids, **kwargs):
        raise AssertionError("WithinUnitKFoldSchema no debería llamar a get_units.")

    def list_units(self):
        raise AssertionError("WithinUnitKFoldSchema no debería llamar a list_units.")


def _build_provider(n=100, seed=0):
    X = np.arange(1, n + 1).reshape(n, 1, 1).astype(float)
    y = np.array([0] * 60 + [1] * 40)
    metadata = pd.DataFrame({"unit": [3] * n})
    return FakeDataProvider(X, y, metadata), X, y


def verify_shared_frozen_test_and_single_provider_call():
    print("--- 1 y 6. Test congelado compartido por los K folds; provider llamado 1 sola vez ---")
    provider, X, y = _build_provider()
    schema = WithinUnitKFoldSchema(provider, k_folds=5, ptest=0.2, seed=123)
    folds = schema.generate_folds(unit_id=3)

    assert len(folds) == 5, f"Se esperaban 5 Fold (k_folds=5), se obtuvieron {len(folds)}"
    assert provider.get_unit_calls == 1, (
        f"provider.get_unit() debería llamarse exactamente 1 vez para generar los 5 folds, "
        f"se llamó {provider.get_unit_calls} veces -- relectura innecesaria por fold."
    )

    test_idx_sets = [set(f.metadata["test_idx"]) for f in folds]
    first = test_idx_sets[0]
    for i, s in enumerate(test_idx_sets[1:], start=1):
        assert s == first, f"El fold {i} tiene test_idx distinto al fold 0 -- el test no está congelado."

    print(f"  provider.get_unit_calls = {provider.get_unit_calls} (esperado: 1)")
    print(f"  test_idx idéntico en los 5 folds ({len(first)} trials)")
    print("  OK\n")
    return folds, y


def verify_no_overlap_and_full_coverage_per_fold(folds, y):
    print("--- 2. Sin overlap y cobertura completa dentro de cada fold ---")
    n = len(y)
    for fold_idx, fold in enumerate(folds):
        train, val, test = set(fold.metadata["train_idx"]), set(fold.metadata["val_idx"]), set(fold.metadata["test_idx"])
        assert not (train & val), f"Fold {fold_idx}: train/val se solapan: {train & val}"
        assert not (train & test), f"Fold {fold_idx}: train/test se solapan: {train & test}"
        assert not (val & test), f"Fold {fold_idx}: val/test se solapan: {val & test}"
        union = train | val | test
        assert union == set(range(n)), (
            f"Fold {fold_idx}: train∪val∪test no cubre exactamente 0..{n-1}.\n"
            f"  faltantes: {set(range(n)) - union}\n  de más: {union - set(range(n))}"
        )
        print(f"  fold {fold_idx}: train={len(train)}, val={len(val)}, test={len(test)}, cobertura completa")
    print("  OK\n")


def verify_folds_differ():
    print("--- 3. Folds distintos tienen particiones train/val distintas ---")
    provider, _, _ = _build_provider()
    schema = WithinUnitKFoldSchema(provider, k_folds=5, ptest=0.2, seed=123)
    folds = schema.generate_folds(unit_id=3)

    train_sets = [frozenset(f.metadata["train_idx"]) for f in folds]
    assert len(set(train_sets)) == len(train_sets), "Dos o más folds tienen exactamente el mismo train_idx"
    print(f"  {len(train_sets)} folds, todos con train_idx distinto entre sí")
    print("  OK\n")


def verify_reproducibility_same_seed():
    print("--- 4. Reproducibilidad: mismo seed -> mismo resultado exacto ---")
    provider_a, _, _ = _build_provider()
    provider_b, _, _ = _build_provider()
    schema_a = WithinUnitKFoldSchema(provider_a, k_folds=4, ptest=0.25, seed=7)
    schema_b = WithinUnitKFoldSchema(provider_b, k_folds=4, ptest=0.25, seed=7)

    folds_a = schema_a.generate_folds(unit_id=1)
    folds_b = schema_b.generate_folds(unit_id=1)

    assert len(folds_a) == len(folds_b)
    for i, (fa, fb) in enumerate(zip(folds_a, folds_b)):
        assert fa.metadata["train_idx"] == fb.metadata["train_idx"], f"Fold {i}: train_idx difiere"
        assert fa.metadata["val_idx"] == fb.metadata["val_idx"], f"Fold {i}: val_idx difiere"
        assert fa.metadata["test_idx"] == fb.metadata["test_idx"], f"Fold {i}: test_idx difiere"
    print(f"  {len(folds_a)} folds, resultado idéntico bit a bit en ambas corridas (seed=7)")
    print("  OK\n")


def verify_indices_point_to_original_X():
    print("--- 5. Los índices apuntan al X original (valores == índice+1) ---")
    provider, X, y = _build_provider()
    X_flat = X.reshape(len(y))
    schema = WithinUnitKFoldSchema(provider, k_folds=3, ptest=0.2, seed=42)
    folds = schema.generate_folds(unit_id=9)

    for fold_idx, fold in enumerate(folds):
        for name, idx_list in [
            ("train", fold.metadata["train_idx"]), ("val", fold.metadata["val_idx"]), ("test", fold.metadata["test_idx"]),
        ]:
            values = X_flat[idx_list]
            expected = np.array(idx_list) + 1
            assert np.array_equal(sorted(values), sorted(expected)), (
                f"Fold {fold_idx}, {name}: valores de X no coinciden con índice+1"
            )
    print(f"  {len(folds)} folds verificados, todos los índices apuntan correctamente al X original")
    print("  OK\n")


def verify_each_pool_trial_is_val_exactly_once(folds, y):
    print("--- 7. Cada trial del pool (train+val) aparece en val_idx de EXACTAMENTE 1 fold ---")
    test_idx = set(folds[0].metadata["test_idx"])
    pool = set(range(len(y))) - test_idx

    val_count = {idx: 0 for idx in pool}
    for fold_idx, fold in enumerate(folds):
        for idx in fold.metadata["val_idx"]:
            assert idx in pool, f"Fold {fold_idx}: val_idx contiene {idx}, que pertenece a test_idx"
            val_count[idx] += 1

    never_val = [idx for idx, count in val_count.items() if count == 0]
    more_than_once = [idx for idx, count in val_count.items() if count > 1]
    assert not never_val, f"{len(never_val)} trials del pool NUNCA aparecen como val: {sorted(never_val)[:10]}..."
    assert not more_than_once, f"{len(more_than_once)} trials del pool aparecen como val MÁS DE UNA VEZ: {sorted(more_than_once)[:10]}..."

    print(f"  pool: {len(pool)} trials, cada uno aparece en val_idx de exactamente 1 fold")
    print("  OK\n")


if __name__ == "__main__":
    folds, y = verify_shared_frozen_test_and_single_provider_call()
    verify_no_overlap_and_full_coverage_per_fold(folds, y)
    verify_folds_differ()
    verify_reproducibility_same_seed()
    verify_indices_point_to_original_X()
    verify_each_pool_trial_is_val_exactly_once(folds, y)
    print("=== TODOS LOS CHECKS DE verify_02_kfold_schema.py PASARON ===")
