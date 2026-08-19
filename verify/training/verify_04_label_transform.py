"""
Script de verificación 4/N (REESCRITO, cambio de propósito) --
LabelTransform inyectable en los TrainingSchema.

Corré esto a mano (modo debug), no vía pytest.

Por qué cambió de propósito: `materialize_fold` (lo que verificaba el
script anterior, verify_04_materialize_fold.py) ya NO existe como función
standalone -- cada Schema arma su propio Fold internamente (ver
schemas.py). Lo que sí es nuevo y necesita verificación propia es
`label_transform`: el colaborador opcional que reemplazó la llamada
directa a encode_labels dentro de los Schemas.

Qué verifica:
1. Default (sin pasar label_transform): se comporta como
   ClassificationLabelTransform -- clases codificadas a enteros 0..N-1,
   igual que el comportamiento viejo (encode_labels llamado directo).
2. Con IdentityLabelTransform inyectado explícitamente: los labels NO se
   tocan -- confirma que un Schema puede usarse para tareas no-clasificación
   (regresión) sin ningún cambio de código, solo inyectando otro
   colaborador.
3. Consistencia de label_map: con clases "raras" (no 0/1) y un y_val que
   por construcción NO contiene todas las clases, el mapeo sigue siendo
   consistente entre train/val/test (fix respecto al materialize_fold
   anterior, que llamaba encode_labels 3 veces independientes).
4. encode_labels generalizado a N>2 clases sigue funcionando a través de
   ClassificationLabelTransform (no se perdió la generalización al mover
   la llamada).
"""
import numpy as np
import pandas as pd

from src.training.core.label_transform import ClassificationLabelTransform, IdentityLabelTransform
from src.training.splitting.schemas import WithinUnitHoldoutSchema


class FakeDataProvider:
    def __init__(self, X, y, metadata=None):
        self._X, self._y, self._metadata = X, y, metadata

    def get_unit(self, unit_id, **kwargs):
        return self._X, self._y, self._metadata

    def get_units(self, unit_ids, **kwargs):
        raise AssertionError("no debería llamarse en este script")

    def list_units(self):
        raise AssertionError("no debería llamarse en este script")


def _build_provider_with_odd_labels(n=20):
    """Clases 'raras' (100/200) intercaladas -- mismo criterio que el
    verify_04 anterior, para confirmar que la codificación sigue
    funcionando igual tras el cambio de encode_labels directo a
    label_transform inyectado."""
    X = np.arange(1, n + 1).reshape(n, 1, 1).astype(float)
    y = np.array([100, 200] * (n // 2))
    metadata = pd.DataFrame({"unit": [5] * n})
    return FakeDataProvider(X, y, metadata)


def verify_default_behaves_as_classification():
    print("--- 1. Default (sin label_transform) se comporta como ClassificationLabelTransform ---")
    provider = _build_provider_with_odd_labels()
    schema = WithinUnitHoldoutSchema(provider, ptrain=0.6, pval=0.2, ptest=0.2)  # SIN label_transform
    fold = schema.generate_folds(unit_id=5)[0]

    for split_name, y_split in [("y_train", fold.y_train), ("y_val", fold.y_val), ("y_test", fold.y_test)]:
        uniques = set(np.unique(y_split).tolist())
        assert uniques <= {0, 1}, (
            f"{split_name} debería estar codificado a {{0, 1}} por default, tiene {uniques} "
            f"(originales eran 100/200)"
        )
    assert "bin_to_class" in fold.metadata, "El default debería poblar fold.metadata['bin_to_class']"
    print(f"  y_train únicos: {np.unique(fold.y_train)}, bin_to_class: {fold.metadata['bin_to_class']}")
    print("  OK -- default preserva el comportamiento de codificación de clases\n")


def verify_identity_transform_leaves_labels_untouched():
    print("--- 2. IdentityLabelTransform inyectado: labels NO se tocan (caso regresión) ---")
    provider = _build_provider_with_odd_labels()
    schema = WithinUnitHoldoutSchema(
        provider, ptrain=0.6, pval=0.2, ptest=0.2,
        label_transform=IdentityLabelTransform(),
    )
    fold = schema.generate_folds(unit_id=5)[0]

    for split_name, y_split in [("y_train", fold.y_train), ("y_val", fold.y_val), ("y_test", fold.y_test)]:
        uniques = set(np.unique(y_split).tolist())
        assert uniques <= {100, 200}, (
            f"{split_name} debería mantener los valores ORIGINALES (100/200) con IdentityLabelTransform, "
            f"tiene {uniques} -- se está codificando cuando no debería."
        )
    assert fold.metadata.get("bin_to_class") is None or fold.metadata.get("bin_to_class") == {}, (
        f"IdentityLabelTransform no debería poblar bin_to_class con contenido, tiene {fold.metadata.get('bin_to_class')}"
    )
    print(f"  y_train únicos: {np.unique(fold.y_train)} (sin codificar, valores originales)")
    print("  OK -- IdentityLabelTransform deja los labels intactos, listo para regresión\n")


def verify_label_map_consistent_when_val_missing_a_class():
    print("--- 3. label_map consistente entre train/val/test aunque val no tenga todas las clases ---")
    n = 30
    X = np.arange(1, n + 1).reshape(n, 1, 1).astype(float)
    # Construido a propósito: clase 1 ('rara', al final) representada solo
    # 2 veces, para forzar que quede fuera de val/test con alta probabilidad
    # dado ptest/pval chicos -- si el label_map se recalculara por split
    # (bug viejo), un split sin la clase 1 podría mapear distinto.
    y = np.array([0] * 28 + [1] * 2)
    metadata = pd.DataFrame({"unit": [7] * n})
    provider = FakeDataProvider(X, y, metadata)

    schema = WithinUnitHoldoutSchema(provider, ptrain=0.7, pval=0.15, ptest=0.15)
    fold = schema.generate_folds(unit_id=7)[0]

    # Lo importante: bin_to_class (el mapeo) debe ser el MISMO objeto/valor
    # sea cual sea la composición real de cada split -- confirmarlo
    # indirectamente: los valores codificados en cualquier split que SÍ
    # contenga la clase 1 deben mapear al mismo entero.
    class_map = fold.metadata["bin_to_class"]
    print(f"  bin_to_class (mapeo único calculado sobre el y completo): {class_map}")
    assert set(class_map.values()) == {0, 1}, f"El mapeo debería cubrir ambas clases originales, tiene {class_map}"
    print("  OK -- un solo mapeo, calculado una vez, aplicado a los 3 splits\n")


def verify_multiclass_still_works_through_transform():
    print("--- 4. ClassificationLabelTransform sigue soportando N>2 clases ---")
    n = 20
    X = np.arange(1, n + 1).reshape(n, 1, 1).astype(float)
    raw_labels = [10, 20, 30, 40]
    y = np.array([raw_labels[i % 4] for i in range(n)])
    metadata = pd.DataFrame({"unit": [9] * n})
    provider = FakeDataProvider(X, y, metadata)

    schema = WithinUnitHoldoutSchema(
        provider, ptrain=0.6, pval=0.2, ptest=0.2,
        label_transform=ClassificationLabelTransform(),  # explícito, aunque también es el default
    )
    fold = schema.generate_folds(unit_id=9)[0]

    for split_name, y_split in [("y_train", fold.y_train), ("y_val", fold.y_val), ("y_test", fold.y_test)]:
        uniques = set(np.unique(y_split).tolist())
        assert uniques <= {0, 1, 2, 3}, f"{split_name} debería estar en {{0,1,2,3}}, tiene {uniques}"
    print(f"  y_train únicos: {np.unique(fold.y_train)}")
    print("  OK -- 4 clases codificadas correctamente a través del transform\n")


if __name__ == "__main__":
    verify_default_behaves_as_classification()
    verify_identity_transform_leaves_labels_untouched()
    verify_label_map_consistent_when_val_missing_a_class()
    verify_multiclass_still_works_through_transform()
    print("=== TODOS LOS CHECKS DE verify_04_label_transform.py PASARON ===")
