"""
`WeightsMatcher` -- contraparte del `TrainingSchema` para el lado testing:
decide qué `TrainedWeights` corresponden a la unidad de test actual. Solo
`WithinUnitMatcher` implementado (lo que hace falta para el caso
within-unit de ablación espectral); matchers para LOSO/cross-dataset
quedan preparados vía registro para agregarse a futuro sin tocar el
orquestador.

Rename `WithinSubjectMatcher` -> `WithinUnitMatcher` (mismo motivo que en
schemas.py: "unit" reemplaza "subject" en todo src/training/, dejando
"subject" únicamente en BCIDataProviderAdapter y eeg_datasets/).

`WithinUnitMatcher` asume el formato de nombrado que generan
`WithinUnitHoldoutSchema`/`WithinUnitKFoldSchema` (unit_XX). Si el
training de origen usó una estrategia con nombrado distinto (ej.
`LeaveOneUnitOutSchema`, que nombra particiones como
`loo_test-unit_XX`), este matcher NO puede interpretarlas -- falla fuerte
(raise) en vez de devolver una lista vacía en silencio, para que el error
señale de entrada que hace falta un Matcher distinto, en vez de aparecer
como "no encontré pesos para esta partición" sin explicación.
"""
import re
from typing import List

from src.training.weights.resolver import TrainedWeights

_UNIT_PARTITION_RE = re.compile(r"^unit_\d+$")


class WithinUnitMatcher:
    def match(self, trained_weights: List[TrainedWeights], test_partition: str) -> List[TrainedWeights]:
        """Todos los TrainedWeights cuya partition coincide con
        test_partition a nivel de unidad (todas las seeds de esa unidad).

        Falla fuerte si test_partition o algún segments[0] de
        trained_weights no tiene el formato unit_XX esperado -- ver
        docstring del módulo."""
        if not _UNIT_PARTITION_RE.match(test_partition):
            raise ValueError(
                f"WithinUnitMatcher espera particiones con formato 'unit_XX', "
                f"recibió '{test_partition}'. Si el training de origen usó una estrategia "
                f"distinta (ej. LeaveOneUnitOutSchema), hace falta un Matcher específico para "
                f"ese formato de nombrado -- ver MATCHER_REGISTRY."
            )

        matched = []
        for w in trained_weights:
            origin_partition = w.metadata["segments"][0]
            if not _UNIT_PARTITION_RE.match(origin_partition):
                raise ValueError(
                    f"WithinUnitMatcher encontró un TrainedWeights con partition de origen "
                    f"'{origin_partition}', que no tiene el formato 'unit_XX' esperado "
                    f"(path: {w.path}). El training de origen probablemente usó una estrategia "
                    f"distinta a WithinUnitHoldoutSchema/WithinUnitKFoldSchema -- hace falta "
                    f"un Matcher específico para ese formato."
                )
            if origin_partition == test_partition:
                matched.append(w)
        return matched


MATCHER_REGISTRY = {
    "WithinUnitMatcher": WithinUnitMatcher,
}