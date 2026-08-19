"""
`LabelTransform` -- colaborador opcional inyectado en cada TrainingSchema,
mismo patrón que `LossStrategy` en Trainer_DL (ver
src/training/execution/loss_strategy.py): default None preserva el
comportamiento actual, un colaborador concreto se activa explícitamente.

Por qué existe: encode_labels (mapeo de clases a enteros 0..N-1) es una
transformación específica de CLASIFICACIÓN -- no tiene sentido para
regresión (y continuo, nada que "codificar") ni para clustering (no hay y
supervisado). Si los Schemas llamaran a encode_labels directo, quedarían
atados a clasificación aunque su algoritmo de partición (holdout,
k-fold, leave-one-out) sea genérico para cualquier tarea. Con
LabelTransform inyectado, el Schema le pide al colaborador "transformá
estos labels", sin saber si eso significa "codificar clases" o "no hacer
nada" (regresión) o cualquier otra cosa a futuro.
"""
from typing import Any, Dict, Optional, Protocol, Tuple

from src.training.utils import encode_labels


class LabelTransform(Protocol):
    def transform(
        self, y_train: Any, y_val: Optional[Any], y_test: Any,
    ) -> Tuple[Any, Optional[Any], Any, Dict[str, Any]]:
        """Devuelve (y_train, y_val, y_test) transformados + un dict de
        metadata para adjuntar a Fold.metadata (ej. {"bin_to_class": ...}
        para clasificación, {} si no aplica). y_val puede ser None (caso
        LeaveOneUnitOutSchema sin val_units) -- las implementaciones deben
        preservar ese None, no asumir que siempre hay val."""
        ...


class ClassificationLabelTransform:
    """Envuelve encode_labels (sin cambios en su lógica interna). Calcula
    el label_map UNA vez sobre y_train (el conjunto más probable de tener
    todas las clases representadas) y lo aplica igual a los 3 splits, para
    que el mapeo sea consistente entre train/val/test incluso si alguno no
    contiene todas las clases -- mismo fix que ya se había identificado
    como pendiente para materialize_fold."""

    def transform(self, y_train, y_val, y_test):
        y_train_enc, label_map = encode_labels(y_train)
        y_val_enc = encode_labels(y_val, label_map=label_map)[0] if y_val is not None else None
        y_test_enc, _ = encode_labels(y_test, label_map=label_map)
        class_map = {v: k for k, v in label_map.items()}
        return y_train_enc, y_val_enc, y_test_enc, {"bin_to_class": class_map}


class IdentityLabelTransform:
    """No-op -- para regresión, o cualquier tarea donde los labels no
    necesitan transformación antes de entrenar."""

    def transform(self, y_train, y_val, y_test):
        return y_train, y_val, y_test, {}