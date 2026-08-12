"""
`TrainedModelResolver` -- agnóstico a semántica de partición (principio 4
del refactor): recorre el árbol de resultados de un training ya corrido y
devuelve TODOS los pesos encontrados, sin asumir qué representa cada nivel
de carpeta más allá de "el primer segmento es la unidad que
`WithinSubjectMatcher` usa para matchear" (eso vive en el matcher, no acá).

Decisión de diseño (glob + filtro por script_progress.csv, no uno u otro
puro): un glob puro no sabe qué particiones fallaron; leer directamente
`script_progress.csv` reintroduce la semántica de path que el principio 4
pide evitar. Acá se hace glob de archivos de pesos y se cruza contra el
`script_progress.csv` del training de origen para descartar particiones
`failed`/`running`, sin asumir nada sobre la profundidad del árbol más allá
de "el primer segmento del path relativo es la partition que aparece en
script_progress.csv" (eso coincide con cómo la escribe
src/training/orchestrator.py).
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

import pandas as pd


@dataclass
class TrainedWeights:
    path: Path
    partition: str  # path relativo (sin el nombre de archivo) tal cual aparece en disco
    metadata: Dict[str, Any] = field(default_factory=dict)


class TrainedModelResolver:
    def __init__(self, results_root: Path, loaded_model_config: dict, weights_glob: str = "best_model.pth"):
        self.root = (
            Path(results_root)
            / loaded_model_config["strategy_name"]
            / loaded_model_config["recipe_name"]
            / loaded_model_config["database_name"]
        )
        self.weights_glob = weights_glob
        self._success_partitions = self._load_success_partitions()

    def _load_success_partitions(self) -> set:
        progress_path = self.root / "script_progress.csv"
        if not progress_path.exists():
            return set()
        df = pd.read_csv(progress_path)
        if df.empty:
            return set()
        latest = df.drop_duplicates(subset="partition", keep="last")
        return set(latest[latest["status"] == "success"]["partition"])

    def resolve_weights(self) -> Iterator[TrainedWeights]:
        if not self.root.exists():
            return
        for path in sorted(self.root.rglob(self.weights_glob)):
            rel = path.relative_to(self.root)
            segments = rel.parts[:-1]  # sin el nombre de archivo
            if not segments:
                continue
            subject_partition = segments[0]
            if subject_partition not in self._success_partitions:
                continue
            yield TrainedWeights(
                path=path,
                partition=str(Path(*segments)),
                metadata={"segments": segments},
            )
