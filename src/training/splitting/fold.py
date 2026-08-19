"""
`Fold` -- único objeto que sobrevive del diseño anterior (Splitter +
PartitionSpec + materialize_fold). Contiene datos reales (X/y por split),
listo para armar DataLoaders/arrays de entrenamiento.

`PartitionSpec` y `materialize_fold` ya NO existen: cada `TrainingSchema`
(ver schemas.py) calcula su partición y arma su propio `Fold` en un solo
paso, sin un objeto intermedio serializable entre medio. La responsabilidad
de "cómo se llega a un Fold" es ahora enteramente del Schema, no de una
función compartida con parámetros condicionales según la estrategia --
ver sesión de diseño para el razonamiento completo.
"""
from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Fold:
    """Resultado final de un TrainingSchema: datos reales, ya particionados
    y con labels codificados, listos para entrenar/testear.

    metadata varía según qué Schema lo generó -- por ejemplo:
      - WithinUnitHoldoutSchema/WithinUnitKFoldSchema: unit_id,
        train_idx/val_idx/test_idx, bin_to_class, metadata_train/val/test.
      - LeaveOneUnitOutSchema: test_unit, train_units, val_units,
        bin_to_class, metadata_train/val/test.
    Cada Schema documenta en su propio código qué keys puebla -- Fold en
    sí no impone estructura sobre metadata más allá de que sea un dict."""
    X_train: Any
    y_train: Any
    X_val: Any
    y_val: Any
    X_test: Any
    y_test: Any
    metadata: Dict[str, Any] = field(default_factory=dict)