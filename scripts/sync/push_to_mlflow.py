"""
Sube a MLFlow toda corrida "terminada" (tiene metrics_results.csv) en
RESULTS_ROOT que todavía no tenga un run en el dashboard. Es
idempotente: usa el archivo marcador .mlflow_run_id para saber qué ya
subió — si corrés esto dos veces, no duplica nada.

Orden del flujo (PROTOCOL.md sección 8):
    fetch_from_server.sh -> push_to_mlflow.py -> build_analytics_db.py

Nota: la lógica de parseo de config/commit es genérica; si tu
run_production.py todavía no deja config.json / .git_commit en cada
carpeta hoja, este script sigue funcionando pero sin esos tags.
"""
import json
import csv
from pathlib import Path

import mlflow

from src.utils.paths import PATHS

MARKER_FILENAME = ".mlflow_run_id"
METRICS_FILENAME = "metrics_results.csv"


def _iter_leaf_run_dirs(results_root: Path):
    """
    Carpetas 'hoja' (context.replicate) con metrics_results.csv, es
    decir, corridas puntuales terminadas. Estructura esperada:
        RESULTS_ROOT/<strategy>/<recipe>/<dataset>/<partition>/<replicate>/
    """
    yield from (p.parent for p in results_root.rglob(METRICS_FILENAME))


def _already_pushed(run_dir: Path) -> bool:
    return (run_dir / MARKER_FILENAME).exists()


def _parse_tags_from_path(run_dir: Path, results_root: Path) -> dict:
    rel = run_dir.relative_to(results_root)
    parts = rel.parts
    if len(parts) < 5:
        raise ValueError(f"Estructura de carpeta inesperada: {run_dir}")
    strategy, recipe, dataset, partition, replicate = parts[:5]
    return {
        "strategy_name": strategy,
        "recipe_name": recipe,
        "context.source": dataset,
        "context.partition": partition,
        "context.replicate": replicate,
    }


def _load_metrics(run_dir: Path) -> list[dict]:
    with open(run_dir / METRICS_FILENAME, newline="") as f:
        return list(csv.DictReader(f))


def _load_reproducibility_extras(run_dir: Path) -> dict:
    """Lee config.json, .git_commit y .docker_image si existen (trío de reproducibilidad)."""
    extra = {}
    config_path = run_dir / "config.json"
    if config_path.exists():
        extra["config"] = json.loads(config_path.read_text())
    commit_path = run_dir / ".git_commit"
    if commit_path.exists():
        extra["git_commit_hash"] = commit_path.read_text().strip()
    docker_path = run_dir / ".docker_image"
    if docker_path.exists():
        extra["docker_image"] = docker_path.read_text().strip()
    return extra


def push_run(run_dir: Path, results_root: Path, project_name: str, experiment_name: str):
    tags = _parse_tags_from_path(run_dir, results_root)
    extra = _load_reproducibility_extras(run_dir)
    metrics_rows = _load_metrics(run_dir)

    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=str(run_dir.relative_to(results_root))) as run:
        mlflow.set_tags({"project_name": project_name, **tags})
        if "git_commit_hash" in extra:
            mlflow.set_tag("git_commit_hash", extra["git_commit_hash"])
        if "docker_image" in extra:
            mlflow.set_tag("docker_image", extra["docker_image"])
        if "config" in extra:
            mlflow.log_dict(extra["config"], "config.json")

        for row in metrics_rows:
            metric_key = row["metric_name"]
            if row.get("split"):
                metric_key = f"{row['split']}_{metric_key}"
            mlflow.log_metric(metric_key, float(row["value"]))

        (run_dir / MARKER_FILENAME).write_text(run.info.run_id)
        print(f"[push_to_mlflow] {run_dir} -> run_id={run.info.run_id}")


def main(project_name: str = "janus-bci", experiment_name: str = "janus-bci"):
    results_root = PATHS.results_root
    pushed = 0
    for run_dir in _iter_leaf_run_dirs(results_root):
        if _already_pushed(run_dir):
            continue
        push_run(run_dir, results_root, project_name, experiment_name)
        pushed += 1
    print(f"[push_to_mlflow] {pushed} corrida(s) nueva(s) subida(s).")


if __name__ == "__main__":
    main()
