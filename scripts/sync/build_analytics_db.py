"""
Reconstruye db/janus_analytics.db DESDE CERO escaneando RESULTS_ROOT.

Fuente de verdad = disco (CSVs). Esta DB nunca se edita a mano ni con
INSERT/UPDATE sueltos: correr este script es la única forma válida de
actualizarla. Por eso sandbox/ solo puede abrirla en modo lectura
(ver db/models.py y sandbox/db_reader.py).

Orden del flujo (PROTOCOL.md sección 8):
    fetch_from_server.sh -> push_to_mlflow.py -> build_analytics_db.py

Solo procesa corridas que YA tienen .mlflow_run_id (es decir, que ya
pasaron por push_to_mlflow.py) — así runs y metrics siempre comparten
el mismo run_id que ves en el dashboard de MLFlow.
"""
import csv
import json
from pathlib import Path

from sqlalchemy.orm import Session

from db.models import Base, Metric, Run, RunFactor, get_engine
from src.utils.paths import PATHS

METRICS_FILENAME = "metrics_results.csv"
MARKER_FILENAME = ".mlflow_run_id"
CONFIG_FILENAME = "config.json"


def parse_replicate(strategy_name: str, replicate: str) -> dict:
    """
    Convierte el string de la carpeta 'replicate' en factores sueltos
    para run_factors. Esta función SÍ depende de la estrategia (el
    formato del string puede variar), pero el resultado siempre cae
    en la misma tabla genérica — agregar una estrategia nueva es
    agregar un caso acá, nunca tocar el esquema de la DB.
    """
    factors = {}
    if strategy_name.startswith("WS"):
        parts = replicate.split("_")
        if replicate.startswith("seed_") and len(parts) == 2:
            # WithinSubjectHoldoutSplitter (src/training/splitters.py): un
            # solo split determinístico por sujeto, repetido por
            # model_init_seed -- ej. seed_399 -> init_seed=399.
            factors["init_seed"] = parts[1]
        elif len(parts) >= 4:
            # formato legado (repo_viejo): split_8_model_399 -> fold=8,
            # init_seed=399.
            factors["fold"] = parts[1]
            factors["init_seed"] = parts[3]
    # TODO: agregar el parseo correspondiente cuando exista, por
    # ejemplo, Across-Dataset (factores: source_dataset, target_dataset)
    # o Across-Subject.
    return factors


def _read_script_type(run_dir: Path, default: str = "train_dl") -> str:
    """Lee `script_type` del config.json de la carpeta hoja (escrito por
    log_reproducibility_trio) -- necesario desde que existen orquestadores
    de test_dl/test_ml además de train_dl/train_ml (antes de este refactor
    todo run era train_dl, por eso el default)."""
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.exists():
        return default
    try:
        return json.loads(config_path.read_text()).get("script_type", default)
    except (json.JSONDecodeError, OSError):
        return default


def build(results_root: Path):
    engine = get_engine(readonly=False)
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    n_runs = 0
    with Session(engine) as session:
        for metrics_csv in results_root.rglob(METRICS_FILENAME):
            run_dir = metrics_csv.parent
            marker = run_dir / MARKER_FILENAME
            if not marker.exists():
                continue  # todavía no pasó por push_to_mlflow.py

            run_id = marker.read_text().strip()
            rel = run_dir.relative_to(results_root)
            strategy, recipe, dataset, partition, replicate = rel.parts[:5]

            run = Run(
                run_id=run_id,
                project_name=results_root.name,  # ajustar si project_name != nombre de carpeta
                strategy_name=strategy,
                recipe_name=recipe,
                script_type=_read_script_type(run_dir),
                context_source=dataset,
                context_partition=partition,
                context_replicate=replicate,
                status="success",
            )
            session.merge(run)

            with open(metrics_csv, newline="") as f:
                for row in csv.DictReader(f):
                    session.add(Metric(
                        run_id=run_id,
                        split=row.get("split"),
                        metric_name=row["metric_name"],
                        value=float(row["value"]),
                    ))

            for factor_name, factor_value in parse_replicate(strategy, replicate).items():
                session.add(RunFactor(
                    run_id=run_id,
                    factor_name=factor_name,
                    factor_value=factor_value,
                ))

            n_runs += 1

        session.commit()

    print(f"[build_analytics_db] reconstruida con {n_runs} run(s) en {PATHS.analytics_db}")


if __name__ == "__main__":
    build(PATHS.results_root)
