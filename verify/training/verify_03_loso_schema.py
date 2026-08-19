"""
Script de verificación 3/N (REESCRITO) -- LeaveOneUnitOutSchema.

Corré esto a mano (modo debug), no vía pytest.

Cambios respecto a la versión anterior (verify_03_loso_splitter.py):
- Cardinalidad nueva: generate_folds(test_unit=X) -- UN identificador, no
  una lista. Antes, LeaveOneSubjectOut.split(subject_list) generaba TODOS
  los folds de una sola llamada; ahora el caller (este script, simulando
  al orquestador) itera y llama una vez por unidad de foco -- mismo
  patrón que los otros 2 Schemas.
- El FakeDataProvider necesita implementar get_units Y list_units además
  de get_unit (los otros 2 Schemas no los necesitaban).
- Se agrega un check de que list_units() se usa para determinar "el
  resto" (no se le pasa nada más al Schema).

Qué verifica (mismos invariantes que antes, con la cardinalidad nueva):
1. Iterando sobre list_units() y llamando generate_folds(test_unit=X) una
   vez por unidad, se cubre cada unidad exactamente una vez como test.
2. REGLA CRÍTICA: sin unidades compartidas entre train/val/test, en cada
   fold individual.
3. Tamaño de val_units respeta pval_subjects (redondeado) sobre las
   unidades restantes.
4. Reproducibilidad: mismo seed -> mismo resultado exacto.
5. Caso límite: pval_subjects=0 -> val_units vacío, val=None en el Fold.
"""
import numpy as np
import pandas as pd

from src.training.splitting.schemas import LeaveOneUnitOutSchema


class FakeDataProvider:
    """A diferencia de los providers de verify_01/02, este SÍ implementa
    get_units y list_units -- son los que LeaveOneUnitOutSchema necesita."""

    def __init__(self, unit_ids):
        self.unit_ids = unit_ids
        self.get_unit_calls = 0
        self.get_units_calls = 0

    def get_unit(self, unit_id, **kwargs):
        self.get_unit_calls += 1
        # Dummy: 5 trials por unidad, valor = unit_id para poder rastrear
        # de dónde vino cada trial en los checks de composición.
        X = np.full((5, 1, 1), fill_value=unit_id, dtype=float)
        y = np.array([0, 1, 0, 1, 0])
        metadata = pd.DataFrame({"unit": [unit_id] * 5})
        return X, y, metadata

    def get_units(self, unit_ids, **kwargs):
        self.get_units_calls += 1
        X_list, y_list, meta_list = [], [], []
        for uid in unit_ids:
            X, y, meta = self.get_unit(uid)
            self.get_unit_calls -= 1  # no contar como llamada individual, get_units es su propio camino
            X_list.append(X)
            y_list.append(y)
            meta_list.append(meta)
        return np.concatenate(X_list), np.concatenate(y_list), pd.concat(meta_list, ignore_index=True)

    def list_units(self):
        return list(self.unit_ids)


def verify_one_fold_per_unit_iterating_list_units():
    print("--- 1. Iterando list_units(), un fold por unidad de test ---")
    unit_ids = list(range(1, 11))
    provider = FakeDataProvider(unit_ids)
    schema = LeaveOneUnitOutSchema(provider, pval_subjects=0.2, seed=0)

    all_folds = []
    for test_unit in provider.list_units():
        folds = schema.generate_folds(test_unit=test_unit)
        assert len(folds) == 1, f"generate_folds(test_unit={test_unit}) debería devolver 1 Fold, devolvió {len(folds)}"
        all_folds.append(folds[0])

    test_units_seen = [f.metadata["test_unit"] for f in all_folds]
    assert sorted(test_units_seen) == sorted(unit_ids), (
        f"Los test_unit de los folds no cubren exactamente unit_ids.\n"
        f"  esperado: {sorted(unit_ids)}\n  obtenido: {sorted(test_units_seen)}"
    )
    print(f"  {len(all_folds)} folds (uno por llamada), cobertura completa de {len(unit_ids)} unidades")
    print("  OK\n")
    return all_folds, unit_ids


def verify_no_unit_shared_between_splits(all_folds, unit_ids):
    print("--- 2. REGLA CRÍTICA: sin unidades compartidas entre train/val/test, cobertura completa ---")
    for fold in all_folds:
        train = set(fold.metadata["train_units"])
        val = set(fold.metadata["val_units"])
        test = {fold.metadata["test_unit"]}

        assert not (train & val), f"REGLA CRÍTICA VIOLADA -- test_unit={fold.metadata['test_unit']}: {train & val} en train Y val"
        assert not (train & test), f"test_unit={fold.metadata['test_unit']}: train y test se solapan"
        assert not (val & test), f"test_unit={fold.metadata['test_unit']}: val y test se solapan"

        union = train | val | test
        assert union == set(unit_ids), (
            f"test_unit={fold.metadata['test_unit']}: train∪val∪test no cubre exactamente unit_ids.\n"
            f"  faltantes: {set(unit_ids) - union}\n  de más: {union - set(unit_ids)}"
        )
    print(f"  {len(all_folds)} folds verificados: ninguna unidad compartida, cobertura completa cada vez")
    print("  OK\n")


def verify_val_size_respects_fraction():
    print("--- 3. Tamaño de val_units respeta pval_subjects sobre las unidades restantes ---")
    unit_ids = list(range(1, 21))
    provider = FakeDataProvider(unit_ids)
    pval_subjects = 0.25
    schema = LeaveOneUnitOutSchema(provider, pval_subjects=pval_subjects, seed=5)

    for test_unit in provider.list_units():
        fold = schema.generate_folds(test_unit=test_unit)[0]
        remaining = len(unit_ids) - 1
        expected_val_size = round(pval_subjects * remaining)
        actual_val_size = len(fold.metadata["val_units"])
        assert actual_val_size == expected_val_size, (
            f"test_unit={test_unit}: val_units tiene {actual_val_size}, se esperaban {expected_val_size}"
        )
    print(f"  todos los folds con |val_units| == round({pval_subjects} * 19) == {round(pval_subjects * 19)}")
    print("  OK\n")


def verify_reproducibility_same_seed():
    print("--- 4. Reproducibilidad: mismo seed -> mismo resultado exacto ---")
    unit_ids = list(range(1, 16))
    provider_a = FakeDataProvider(unit_ids)
    provider_b = FakeDataProvider(unit_ids)
    schema_a = LeaveOneUnitOutSchema(provider_a, pval_subjects=0.3, seed=99)
    schema_b = LeaveOneUnitOutSchema(provider_b, pval_subjects=0.3, seed=99)

    for test_unit in unit_ids:
        fold_a = schema_a.generate_folds(test_unit=test_unit)[0]
        fold_b = schema_b.generate_folds(test_unit=test_unit)[0]
        assert sorted(fold_a.metadata["train_units"]) == sorted(fold_b.metadata["train_units"]), (
            f"train_units difiere entre corridas con mismo seed para test_unit={test_unit}"
        )
        assert sorted(fold_a.metadata["val_units"]) == sorted(fold_b.metadata["val_units"]), (
            f"val_units difiere entre corridas con mismo seed para test_unit={test_unit}"
        )
    print(f"  {len(unit_ids)} folds, resultado idéntico en ambas corridas (seed=99)")
    print("  OK\n")


def verify_zero_val_fraction_edge_case():
    print("--- 5. Caso límite: pval_subjects=0 -> val_units vacío, X_val/y_val = None ---")
    unit_ids = [10, 20, 30, 40, 50]
    provider = FakeDataProvider(unit_ids)
    schema = LeaveOneUnitOutSchema(provider, pval_subjects=0.0, seed=1)

    for test_unit in unit_ids:
        fold = schema.generate_folds(test_unit=test_unit)[0]
        assert fold.metadata["val_units"] == [], f"val_units debería estar vacío, es {fold.metadata['val_units']}"
        assert fold.X_val is None and fold.y_val is None, (
            f"Con val_units vacío, Fold.X_val/y_val deberían ser None, son "
            f"X_val={fold.X_val is not None}, y_val={fold.y_val is not None}"
        )
        assert sorted(fold.metadata["train_units"]) == sorted([u for u in unit_ids if u != test_unit]), (
            "Con val vacío, train_units debería ser TODAS las unidades restantes"
        )
    print(f"  {len(unit_ids)} folds, val_units vacío y Fold.X_val/y_val = None en todos")
    print("  OK\n")


if __name__ == "__main__":
    all_folds, unit_ids = verify_one_fold_per_unit_iterating_list_units()
    verify_no_unit_shared_between_splits(all_folds, unit_ids)
    verify_val_size_respects_fraction()
    verify_reproducibility_same_seed()
    verify_zero_val_fraction_edge_case()
    print("=== TODOS LOS CHECKS DE verify_03_loso_schema.py PASARON ===")
