"""
Script de verificación 1/N (REESCRITO para la API de TrainingSchema) --
WithinUnitHoldoutSchema / stratified_sequential_split.

Corré esto a mano (modo debug), no vía pytest.

Cambios respecto a la versión anterior (verify_01_holdout_splitter.py,
API vieja Splitter.split(X, y, ...)):
- Ya no se llama a X/y directo -- se arma un FakeDataProvider (implementa
  SOLO get_unit, que es lo único que WithinUnitHoldoutSchema necesita) y
  se instancia el Schema con provider=fake_provider.
- generate_folds(unit_id=...) reemplaza a split(X, y, subject_id=...).
- Se agrega un check nuevo (5): el provider se llama EXACTAMENTE UNA VEZ
  por invocación de generate_folds -- antes esto se resolvía con el atajo
  opcional X=/y= de materialize_fold; ahora que ese atajo no existe (el
  Schema siempre pide los datos al provider), lo que hay que confirmar es
  que no se llame de más (por ejemplo, dos veces por error entre el split
  de test y el split de val).

Qué verifica (mismos invariantes que la versión anterior, más el nuevo):
1. stratified_sequential_split, con clases DESBALANCEADAS y ntest no
   divisible exactamente -- floor-division, caso documentado como
   intencional.
2. train/test sin overlap (caso balanceado).
3. WithinUnitHoldoutSchema.generate_folds() devuelve List[Fold] de
   longitud 1.
4. FIX de índices: train_idx/val_idx/test_idx en el MISMO espacio de
   referencia (el X/y original), verificado con cálculo manual exacto del
   algoritmo de dos pasos, más lectura directa de valores usando un
   vector secuencial 1..100.
5. NUEVO: el provider.get_unit() se llama exactamente 1 vez por llamada a
   generate_folds() -- no se relee de más.
"""
import numpy as np
import pandas as pd

from src.training.splitting.schemas import stratified_sequential_split, WithinUnitHoldoutSchema


class FakeDataProvider:
    """Implementa SOLO get_unit -- lo único que WithinUnitHoldoutSchema
    necesita del contrato DataProvider. Cuenta llamadas para el check 5."""

    def __init__(self, X, y, metadata=None):
        self._X, self._y, self._metadata = X, y, metadata
        self.get_unit_calls = 0

    def get_unit(self, unit_id, **kwargs):
        self.get_unit_calls += 1
        return self._X, self._y, self._metadata

    def get_units(self, unit_ids, **kwargs):
        raise AssertionError("WithinUnitHoldoutSchema no debería llamar a get_units (es within-unit, no multi-unit).")

    def list_units(self):
        raise AssertionError("WithinUnitHoldoutSchema no debería llamar a list_units (no necesita saber qué otras unidades hay).")


def verify_stratified_sequential_split_floor_division():
    print("--- 1. stratified_sequential_split: floor-division con clases desbalanceadas ---")
    labels = np.array([0] * 13 + [1] * 7)
    train_idx, test_idx = stratified_sequential_split(ntest=5, labels=labels)

    expected_train_idx = list(range(0, 7)) + list(range(13, 20))
    expected_test_idx = list(range(7, 9))
    excluded = set(range(9, 13))

    assert sorted(train_idx) == sorted(expected_train_idx), (
        f"train_idx no coincide.\n  esperado: {sorted(expected_train_idx)}\n  obtenido: {sorted(train_idx)}"
    )
    assert sorted(test_idx) == sorted(expected_test_idx), (
        f"test_idx no coincide.\n  esperado: {sorted(expected_test_idx)}\n  obtenido: {sorted(test_idx)}"
    )
    assert excluded.isdisjoint(set(train_idx)) and excluded.isdisjoint(set(test_idx)), (
        f"Índices {excluded} deberían quedar excluidos por floor-division"
    )
    print(f"  train_idx: {sorted(train_idx)}")
    print(f"  test_idx: {sorted(test_idx)}")
    print(f"  excluidos por floor-division: {sorted(excluded)}")
    print("  OK\n")


def verify_train_test_no_overlap():
    print("--- 2. train_idx / test_idx sin overlap (caso balanceado) ---")
    labels = np.array([0] * 20 + [1] * 20)
    train_idx, test_idx = stratified_sequential_split(ntest=10, labels=labels)
    overlap = set(train_idx) & set(test_idx)
    assert not overlap, f"train_idx y test_idx se solapan en: {overlap}"
    print(f"  train_idx: {len(train_idx)} elementos, test_idx: {len(test_idx)} elementos, sin overlap")
    print("  OK\n")


def _manual_stratified_sequential_split(ntest, labels):
    """Reimplementación independiente, para comparar sin depender del
    propio código que se está verificando."""
    n = len(labels)
    ntrain = n - ntest
    unique_labels = np.unique(labels)
    ntrain_per_class = int(ntrain // len(unique_labels))
    ntest_per_class = int(ntest // len(unique_labels))

    train_idx, test_idx = [], []
    for cls in unique_labels:
        cls_indices = np.where(labels == cls)[0]
        train_idx.extend(cls_indices[:ntrain_per_class])
        test_idx.extend(cls_indices[ntrain_per_class:ntrain_per_class + ntest_per_class])
    return train_idx, test_idx


def verify_schema_with_sequential_vector():
    print("--- 3, 4 y 5. WithinUnitHoldoutSchema con X/y secuencial 1..100, vía FakeDataProvider ---")

    n = 100
    X = np.arange(1, n + 1).reshape(n, 1, 1).astype(float)
    y = np.array([0] * 65 + [1] * 35)
    metadata = pd.DataFrame({"unit": [42] * n})
    provider = FakeDataProvider(X, y, metadata)

    ptrain, pval, ptest = 0.6, 0.2, 0.2
    schema = WithinUnitHoldoutSchema(provider, ptrain=ptrain, pval=pval, ptest=ptest)
    folds = schema.generate_folds(unit_id=42)

    assert isinstance(folds, list) and len(folds) == 1, (
        f"generate_folds debe devolver List[Fold] de longitud 1, devolvió {type(folds)} "
        f"de longitud {len(folds) if isinstance(folds, list) else 'N/A'}"
    )
    fold = folds[0]
    assert fold.metadata.get("unit_id") == 42, f"metadata['unit_id'] debería ser 42, es {fold.metadata.get('unit_id')}"

    # Check 5: el provider se llamó exactamente 1 vez.
    assert provider.get_unit_calls == 1, (
        f"provider.get_unit() debería haberse llamado exactamente 1 vez, se llamó "
        f"{provider.get_unit_calls} veces -- posible relectura innecesaria dentro de generate_folds."
    )

    # Check 4: replicar el algoritmo completo a mano.
    expected_train_idx_intermediate, expected_test_idx = _manual_stratified_sequential_split(
        ntest=ptest * n, labels=y
    )
    y_train_intermediate = y[expected_train_idx_intermediate]
    expected_train_idx_relative, expected_val_idx_relative = _manual_stratified_sequential_split(
        ntest=pval * n, labels=y_train_intermediate
    )
    expected_train_idx = [expected_train_idx_intermediate[i] for i in expected_train_idx_relative]
    expected_val_idx = [expected_train_idx_intermediate[i] for i in expected_val_idx_relative]

    assert sorted(fold.metadata["test_idx"]) == sorted(expected_test_idx), (
        f"test_idx no coincide con el cálculo manual.\n"
        f"  esperado: {sorted(expected_test_idx)}\n  obtenido: {sorted(fold.metadata['test_idx'])}"
    )
    assert sorted(fold.metadata["train_idx"]) == sorted(expected_train_idx), (
        f"train_idx no coincide con el cálculo manual.\n"
        f"  esperado: {sorted(expected_train_idx)}\n  obtenido: {sorted(fold.metadata['train_idx'])}"
    )
    assert sorted(fold.metadata["val_idx"]) == sorted(expected_val_idx), (
        f"val_idx no coincide con el cálculo manual.\n"
        f"  esperado: {sorted(expected_val_idx)}\n  obtenido: {sorted(fold.metadata['val_idx'])}"
    )

    # Verificación "a ojo": los VALORES de X_train/X_val/X_test deben ser
    # exactamente (índice+1), confirmando que apuntan al X original.
    for name, X_split, idx_list in [
        ("train", fold.X_train, fold.metadata["train_idx"]),
        ("val", fold.X_val, fold.metadata["val_idx"]),
        ("test", fold.X_test, fold.metadata["test_idx"]),
    ]:
        values = X_split.reshape(-1)
        expected_values = np.array(idx_list) + 1
        assert np.array_equal(sorted(values), sorted(expected_values)), (
            f"Los valores de X_{name} no coinciden con índice+1 -- el indexado no apunta al X original."
        )

    # Sin overlap entre los tres, en el espacio original.
    sets = {"train": set(fold.metadata["train_idx"]), "val": set(fold.metadata["val_idx"]), "test": set(fold.metadata["test_idx"])}
    for name_a, set_a in sets.items():
        for name_b, set_b in sets.items():
            if name_a >= name_b:
                continue
            assert not (set_a & set_b), f"{name_a}_idx y {name_b}_idx se solapan: {set_a & set_b}"

    print(f"  provider.get_unit_calls = {provider.get_unit_calls} (esperado: 1)")
    print(f"  test_idx  ({len(fold.metadata['test_idx'])}): {sorted(fold.metadata['test_idx'])}")
    print(f"  train_idx ({len(fold.metadata['train_idx'])}): {sorted(fold.metadata['train_idx'])}")
    print(f"  val_idx   ({len(fold.metadata['val_idx'])}): {sorted(fold.metadata['val_idx'])}")
    print("  Coincide exacto con el cálculo manual. Valores de X confirmados contra el original.")
    print("  OK\n")


if __name__ == "__main__":
    verify_stratified_sequential_split_floor_division()
    verify_train_test_no_overlap()
    verify_schema_with_sequential_vector()
    print("=== TODOS LOS CHECKS DE verify_01_holdout_schema.py PASARON ===")
