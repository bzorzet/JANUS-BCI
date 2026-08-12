"""
`WeightsMatcher` -- contraparte del `Splitter` para el lado testing: decide
qué `TrainedWeights` corresponden a la partición de test actual. Solo
`WithinSubjectMatcher` en este refactor (lo que hace falta para el caso
within-subject de ablación espectral); `AcrossDatasetMatcher` queda
preparado vía registro para agregarse a futuro sin tocar el orquestador.
"""
from typing import List

from src.training.weights_resolver import TrainedWeights


class WithinSubjectMatcher:
    def match(self, trained_weights: List[TrainedWeights], test_partition: str) -> List[TrainedWeights]:
        """Todos los TrainedWeights cuya partition coincide con
        test_partition a nivel de sujeto (todas las seeds de ese sujeto)."""
        return [w for w in trained_weights if w.metadata["segments"][0] == test_partition]


MATCHER_REGISTRY = {
    "WithinSubjectMatcher": WithinSubjectMatcher,
}
