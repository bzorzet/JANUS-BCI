"""
Orquestador de testing (prompt maestro sección 8). Evalúa modelos YA
entrenados -- posiblemente con un dataset/dataloader de ablación distinto
al usado en training (ej. MultiFrequencyBandMaskingDataset para ablación
espectral) -- contra el `strategy_name`/`recipe_name`/`database_name`
propios de ESTE config de test, que pueden ser distintos de los del
training de origen (`loaded_model_config`).

Simplificación deliberada respecto al script de testing viejo: se evalúa
UN solo `test_loader` construido sobre `dataset.flatten_subject_data(...)`
completo (no se reconstruyen los splits train/val/test de la corrida de
origen vía `info_data_training.json` -- ese mecanismo era específico de
los índices que guardaba el training script viejo). Ya reflejado así en el
plan aprobado (sección 8.3): una fila de métricas por seed emparejada, sin
agregación, split único "test".
"""
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.torch_utils import obtain_cuda_device
from src.training.orchestrator import _flatten_for_mlflow, make_run_dir
from src.training.tester import Tester_DL
from src.training.utils import binarize_y, create_dataloader, create_dataset
from src.training.weights_matcher import MATCHER_REGISTRY
from src.training.weights_resolver import TrainedModelResolver
from src.utils.imports import import_class
from src.utils.paths import PATHS
from src.utils.reproducibility import log_reproducibility_trio
from src.utils.run_lifecycle import RunLifecycleManager, _append_csv_row

logger = logging.getLogger(__name__)


def _testing_identity(config: dict) -> dict:
    identity = copy.deepcopy(config)
    general = identity.get("general_script_config", {})
    general.pop("device", None)
    return identity


def _compute_test_metrics(y_true, y_pred, y_pred_proba) -> List[Dict[str, Any]]:
    """Mismo set de métricas que el training (accuracy/auc/recall/precision
    por clase), en un único split "test" -- acá no hay train/val, solo la
    evaluación del modelo cargado contra los datos de test de este config."""
    rows: List[Dict[str, Any]] = [
        {"split": "test", "metric_name": "accuracy", "value": accuracy_score(y_true, y_pred)},
    ]
    auc = roc_auc_score(y_true, y_pred_proba[:, 1]) if len(np.unique(y_true)) > 1 else np.nan
    rows.append({"split": "test", "metric_name": "auc", "value": auc})
    for cls in (0, 1):
        rows.append({
            "split": "test", "metric_name": f"recall_{cls}",
            "value": recall_score(y_true, y_pred, pos_label=cls, zero_division=0),
        })
        rows.append({
            "split": "test", "metric_name": f"precision_{cls}",
            "value": precision_score(y_true, y_pred, pos_label=cls, zero_division=0),
        })
    return rows


def run_testing_sweep(
    config: dict,
    docker_image: Optional[str],
    reporter,
    force: bool = False,
    overwrite_existing: bool = False,
) -> None:
    project_config = config["project_config"]
    strategy_name = project_config["strategy_name"]
    recipe_name = project_config["model_to_test"]
    database_name = project_config["database_name"]
    database_session = project_config["database_session"]

    loaded_model_config = config["loaded_model_config"]
    general_config = config.get("general_script_config", {})

    config_snapshot = copy.deepcopy(config)

    output_path = PATHS.results_root / strategy_name / recipe_name / database_name
    output_path.mkdir(parents=True, exist_ok=True)

    lifecycle = RunLifecycleManager(
        output_path,
        progress_filename="script_progress.csv",
        identity_filename="config.json",
        force=force,
        overwrite_existing=overwrite_existing,
        emit_running_row=False,
    )
    lifecycle.check_identity_or_raise(
        new_config=config_snapshot,
        extract_identity=_testing_identity,
        extra_cleanup=lambda: [p.unlink() for p in output_path.rglob("*.csv")],
        context_label=f"la corrida de testing en {output_path}",
    )
    log_reproducibility_trio(output_path, config_snapshot, docker_image)

    dataset_cfg = config["databases"]
    dataset = import_class(dataset_cfg["class_name"], dataset_cfg["module_name"])(**dataset_cfg["params"])

    sessions = getattr(dataset, "sessions", [])
    if database_session not in sessions:
        raise ValueError(f"Session {database_session} not found in dataset. Available: {sessions}")

    # A diferencia de training, acá siempre se evalúan todos los sujetos
    # del dataset (no hay subjects_to_test) -- el filtro real es "¿tiene
    # este sujeto pesos entrenados emparejados?", resuelto abajo vía el
    # resolver + matcher.
    subjects_id = dataset.subject_list

    resolver = TrainedModelResolver(PATHS.results_root, loaded_model_config)
    all_weights = list(resolver.resolve_weights())
    matcher = MATCHER_REGISTRY["WithinSubjectMatcher"]()

    partition_to_subject = {f"subject_{s:02d}": s for s in subjects_id}
    matches_by_partition: Dict[str, list] = {}
    for partition in partition_to_subject:
        matched = matcher.match(all_weights, partition)
        if matched:
            matches_by_partition[partition] = matched

    if not matches_by_partition:
        logger.info(
            "Ningún sujeto de %s tiene pesos entrenados emparejados bajo %s -- nada para hacer.",
            database_name, loaded_model_config,
        )
        return

    def _still_present(partition: str) -> bool:
        subject_dir = output_path / partition
        if not subject_dir.exists():
            return False
        return all(
            (subject_dir / w.metadata["segments"][1] / "metrics_results.csv").exists()
            for w in matches_by_partition[partition]
        )

    pending = lifecycle.prepare_partitions(list(matches_by_partition.keys()), still_present=_still_present)

    if not pending:
        logger.info("Todo ya procesado (%d particiones) -- nada para hacer.", len(matches_by_partition))
        return

    device = obtain_cuda_device()
    g = torch.Generator()
    g.manual_seed(general_config.get("seed", 0))

    reporter.on_start(len(pending), meta={
        "strategy_name": strategy_name, "recipe_name": recipe_name, "database_name": database_name,
        "n_partitions": len(pending),
    })

    mlflow.set_experiment(f"{strategy_name}/{recipe_name}/{database_name}")

    for partition in pending:
        subject_id = partition_to_subject[partition]
        ts_start = lifecycle.mark_start(partition)
        reporter.on_run_start(partition)
        partition_status = "success"
        try:
            X, y, metadata = dataset.flatten_subject_data(subject_id, session=database_session)
            y, bin_to_class = binarize_y(y)

            X_t = torch.tensor(X, dtype=torch.float).to(device)
            y_t = torch.tensor(y, dtype=torch.long).to(device)

            test_dataset = create_dataset("test", X_t, y_t, config["torch_dataset"])
            test_loader = create_dataloader("test", test_dataset, config["torch_dataloader"], g)
            y_true = y_t.cpu().numpy()

            for weights in matches_by_partition[partition]:
                origin_replicate = weights.metadata["segments"][1]
                run_dir = make_run_dir(strategy_name, recipe_name, database_name, partition, origin_replicate)
                try:
                    model_cfg = config["model"]
                    model = import_class(model_cfg["class_name"], model_cfg["module_name"])(**model_cfg["params"])
                    model.load_state_dict(torch.load(weights.path, map_location=device))
                    model.to(device)

                    criterion = None
                    criterion_cfg = config.get("criterion")
                    if criterion_cfg:
                        criterion = import_class(criterion_cfg["class_name"], criterion_cfg["module_name"])(**criterion_cfg["params"])

                    tags = {
                        "project_name": project_config.get("project_name") or "",
                        "strategy_name": strategy_name,
                        "database_name": database_name,
                        "database_session": database_session,
                        "label": project_config.get("label") or "",
                        "subject_id": str(subject_id),
                        "origin_partition": weights.partition,
                    }
                    with mlflow.start_run(run_name=f"{partition}/{origin_replicate}", tags=tags) as run:
                        mlflow.log_params(_flatten_for_mlflow(config_snapshot))

                        tester = Tester_DL(model, loss_fn=criterion)
                        y_pred_proba, loss = tester.test(test_loader, probability=True)
                        y_pred_proba = y_pred_proba.cpu().numpy()
                        y_pred = np.argmax(y_pred_proba, axis=1)

                        log_reproducibility_trio(run_dir, config_snapshot, docker_image)

                        metrics_rows = _compute_test_metrics(y_true, y_pred, y_pred_proba)
                        metrics_path = run_dir / "metrics_results.csv"
                        for row in metrics_rows:
                            _append_csv_row(metrics_path, row)
                            mlflow.log_metric(row["metric_name"], row["value"])
                        if loss is not None:
                            mlflow.log_metric("loss", float(loss))

                        (run_dir / ".mlflow_run_id").write_text(run.info.run_id)

                    reporter.on_run_result(f"{partition}/{origin_replicate}", "success")
                except Exception as e:
                    logger.error("Fallo testeando %s/%s: %s", partition, origin_replicate, e)
                    partition_status = "failed"
                    reporter.on_run_result(f"{partition}/{origin_replicate}", "failed")

        except Exception as e:
            logger.error("Fallo cargando/evaluando %s: %s", partition, e)
            partition_status = "failed"

        lifecycle.mark_result(partition, partition_status, ts_start)

    logger.info("--- Testing Finished ---")
    reporter.on_finish({"total": len(pending)})
