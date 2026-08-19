"""
`MetricsStrategy` -- colaborador configurable por JSON (mismo patrón
import_class que model/optimizer/schema), reemplaza _compute_test_metrics
hardcodeada dentro de orchestrator_testing.py.

Por qué existe: _compute_test_metrics tenía roc_auc_score(y_true,
y_pred_proba[:, 1]) (columna 1 hardcodeada) y un loop for cls in (0, 1)
-- ambos rotos silenciosamente con N>2 clases (AUC incompleto/incorrecto,
clases 2+ ignoradas sin error ni warning). Migrado acá y generalizado a N
clases.
"""
from typing import Any, Dict, List, Optional, Protocol


class MetricsStrategy(Protocol):
    def compute(
        self, y_true, y_pred, y_pred_proba=None, split: str = "test",
    ) -> List[Dict[str, Any]]:
        """Devuelve filas listas para metrics_results.csv:
        [{"split":..., "metric_name":..., "value":...}]"""
        ...