"""
Punto de entrada único para una corrida de producción (un barrido:
todas las partitions de un recipe+dataset). Esto es lo que lo
diferencia de sandbox/: todo lo que pasa por acá queda trackeado.

Recibe un --config <path> y despacha según config["script_type"]
(PROTOCOL.md sección 2 — el campo es obligatorio en todo config, sección
5). Un dominio nuevo (o un script_type nuevo) se agrega registrando una
entrada en DISPATCH, no reescribiendo este archivo.

Es un esqueleto para train_dl/train_ml: la lógica específica de tu
dominio (BCI, imágenes, PINNs) vive en src/training o src/analysis, no
acá. Preprocessing ya está completo — delega directo a
EEGDatabasePreprocessor (src/preprocessing/database_preprocessor.py).
"""
import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import EEGDatabasePreprocessor, ProgressReporter, RichProgressReporter
from src.training.orchestrator import run_training_sweep
from src.training.orchestrator_testing import run_testing_sweep
from src.utils.imports import import_class
from src.utils.paths import PATHS


def make_run_dir(strategy: str, recipe: str, dataset: str, partition: str, replicate: str) -> Path:
    run_dir = PATHS.results_root / strategy / recipe / dataset / partition / replicate
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def run_preprocessing(
    config: dict,
    docker_image: Optional[str],
    reporter: ProgressReporter,
    force: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """script_type == 'preprocessing'. El trío de reproducibilidad y el
    manifest los escribe el propio EEGDatabasePreprocessor.run()."""
    db_config = config["database"]
    dataset_cls = import_class(db_config["class_name"], db_config["module_name"])
    dataset = dataset_cls(**db_config["kwargs"])

    orchestrator = EEGDatabasePreprocessor(
        dataset, config["preprocessing_name"], config, docker_image=docker_image, reporter=reporter,
        force=force, overwrite_existing=overwrite_existing,
    )
    orchestrator.run()


def run_training(
    config: dict,
    docker_image: Optional[str],
    reporter: ProgressReporter,
    force: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """script_type == 'train_dl' / 'train_ml'. Delega a
    src.training.orchestrator.run_training_sweep -- este archivo es un
    entrypoint delgado, la lógica de dominio vive en src/training/."""
    run_training_sweep(config, docker_image, reporter, force=force, overwrite_existing=overwrite_existing)


def run_testing(
    config: dict,
    docker_image: Optional[str],
    reporter: ProgressReporter,
    force: bool = False,
    overwrite_existing: bool = False,
) -> None:
    """script_type == 'test_dl' / 'test_ml'. Delega a
    src.training.orchestrator_testing.run_testing_sweep -- evalúa modelos ya
    entrenados (posiblemente con un dataset/ablación distinto al de
    training), no reentrena nada."""
    run_testing_sweep(config, docker_image, reporter, force=force, overwrite_existing=overwrite_existing)


DISPATCH = {
    "preprocessing": run_preprocessing,
    "train_dl": run_training,
    "train_ml": run_training,
    "test_dl": run_testing,
    "test_ml": run_testing,
}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to json config file",
                        default="/home/bzorzet/JANUS-BCI/preprocessing/configs/Cho2017/Cho2017_s1_simple-CAR-128Hz-preproc.json")
    
    parser.add_argument("--docker-image", default=None)
    
    parser.add_argument("--progress", choices=["none", "rich"], default="rich",
                        help="Progreso en terminal: 'rich' para tabla+barra en vivo (debug interactivo), 'none' headless.")
    
    parser.add_argument("--force", action="store_true",
                        help="Ignora run_registry.csv y reprocesa todo, incluso lo ya marcado success.")
    
    parser.add_argument("--overwrite-existing", action="store_true",
                        help="Si el preprocessing_pipeline guardado difiere del actual, sobrescribe "
                             "config.json/.git_commit/.docker_image y reinicia run_registry.csv, "
                             "stage_metrics.csv y las detail tables en vez de rechazar.")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())

    script_type = config.get("script_type")
    handler = DISPATCH.get(script_type)
    if handler is None:
        raise ValueError(
            f"script_type '{script_type}' ausente o desconocido en {args.config}. "
            f"Soportados: {sorted(DISPATCH)} (PROTOCOL.md sección 2) — es un campo "
            "obligatorio explícito en todo config (PROTOCOL.md sección 5)."
        )
    reporter = RichProgressReporter() if args.progress == "rich" else ProgressReporter()
    handler(config, args.docker_image, reporter, force=args.force, overwrite_existing=args.overwrite_existing)


if __name__ == "__main__":
    main()
