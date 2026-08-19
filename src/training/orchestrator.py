"""
Orquestador de training (prompt maestro sección 7). Único componente que
conoce el JSON de config completo -- Splitter/Trainer_DL/CallbackDispatcher
reciben todo ya resuelto (principio 2 del refactor).

Migra el flujo de
repo_viejo/train_model_within_subject_online_simulated_for_MI-BCI_classification.py,
reorganizado: split + datasets/dataloaders se arman UNA VEZ por sujeto (el
split es determinístico, no depende de seed -- ver splitters.py); el loop
sobre `general_script_config.model_init_seed` (una repetición == un
`replicate`) vive acá, no en el Splitter.

Fix ya aprobado con el usuario: el loader de evaluación final post-training
se arma desde una COPIA de `config["torch_dataloader"]`, nunca mutando el
dict de config compartido por todo el barrido (el script viejo mutaba
`shuffle=False` in place, filtrando ese cambio silenciosamente hacia el
training real de la siguiente iteración).
"""
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

from src.torch_utils import EpochScoring, LossTracker, TimerCallback, obtain_cuda_device, set_seed
from src.training.tracking.callbacks_mlflow import MLflowCallback
from src.training.splitting.splitters import PartitionSpec, WithinSubjectHoldoutSplitter, materialize_fold
from src.training.core.trainer import Trainer_DL
from src.training.utils import accuracy, build_callbacks, create_dataloader, create_dataset
from src.utils.imports import import_class
from src.utils.paths import PATHS
from src.utils.reproducibility import log_reproducibility_trio
from src.utils.run_lifecycle import RunLifecycleManager, _append_csv_row

logger = logging.getLogger(__name__)


def make_run_dir(strategy: str, recipe: str, dataset: str, partition: str, replicate: str) -> Path:
    """Mismo layout que scripts/run_production.py::make_run_dir -- se
    re-declara acá para que src/training no dependa de scripts/ (los
    scripts son entrypoints delgados que importan de src/, no al revés)."""
    run_dir = PATHS.results_root / strategy / recipe / dataset / partition / replicate
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def partition_name(spec: PartitionSpec) -> str:
    """Deriva el nombre de partición desde PartitionSpec.metadata en vez de
    asumir un formato fijo -- distinto Splitter, distinto metadata (ver
    src/training/splitters.py, sección 5 del prompt de diseño)."""
    if "test_subject" in spec.metadata:
        name = f"loso_test-subject_{spec.metadata['test_subject']:02d}"
    else:
        name = f"subject_{spec.metadata['subject_id']:02d}"
    if "fold_idx" in spec.metadata:
        name = f"{name}_fold_{spec.metadata['fold_idx']}"
    return name


def _training_identity(config: dict) -> dict:
    """Identidad de una corrida de training para RunLifecycleManager: el
    config completo, salvo campos volátiles que no cambian qué se está
    entrenando (`device` es un detalle de la máquina, `subjects_to_test`
    solo filtra qué particiones correr en ESTA invocación puntual)."""
    identity = copy.deepcopy(config)
    general = identity.get("general_script_config", {})
    general.pop("device", None)
    general.pop("subjects_to_test", None)
    return identity


def _flatten_for_mlflow(config: dict, parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    flat: Dict[str, Any] = {}
    for key, value in config.items():
        full_key = f"{parent_key}{sep}{key}" if parent_key else key
        if isinstance(value, dict):
            flat.update(_flatten_for_mlflow(value, full_key, sep))
        elif isinstance(value, list):
            flat[full_key] = str(value)
        else:
            flat[full_key] = value
    return flat


def _write_train_curve_csv(path: Path, history: Dict[str, Any]) -> None:
    """`train_curve.csv` -- opcional, por época (PROTOCOL.md sección 4).
    Sin contrato de columnas fijo en PROTOCOL.md sección 6 (ese contrato
    solo cubre script_progress.csv/metrics_results.csv) -- una fila por
    época, una columna por cada serie per-época de trainer.history."""
    epochs = history.get("epoch", [])
    per_epoch_keys = [k for k, v in history.items() if isinstance(v, list) and k != "epoch"]
    rows = []
    for i, epoch in enumerate(epochs):
        row: Dict[str, Any] = {"epoch": epoch}
        for key in per_epoch_keys:
            values = history[key]
            if i < len(values):
                row[key] = values[i]
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def _compute_final_metrics(y_train, y_val, y_test, proba_train, proba_val, proba_test) -> List[Dict[str, Any]]:
    """Puerto verbatim del bloque de métricas finales del script viejo:
    accuracy/auc/recall_0/recall_1/precision_0/precision_1 por split, en
    formato largo (PROTOCOL.md sección 6: split/metric_name/value)."""
    rows: List[Dict[str, Any]] = []
    for split_name, y_true_tensor, proba in (
        ("train", y_train, proba_train), ("val", y_val, proba_val), ("test", y_test, proba_test),
    ):
        y_true = y_true_tensor.cpu().numpy()
        y_pred = np.argmax(proba, axis=1)
        rows.append({"split": split_name, "metric_name": "accuracy", "value": accuracy_score(y_true, y_pred)})
        auc = roc_auc_score(y_true, proba[:, 1]) if len(np.unique(y_true)) > 1 else np.nan
        rows.append({"split": split_name, "metric_name": "auc", "value": auc})
        for cls in (0, 1):
            rows.append({
                "split": split_name, "metric_name": f"recall_{cls}",
                "value": recall_score(y_true, y_pred, pos_label=cls, zero_division=0),
            })
            rows.append({
                "split": split_name, "metric_name": f"precision_{cls}",
                "value": precision_score(y_true, y_pred, pos_label=cls, zero_division=0),
            })
    return rows


def run_training_sweep(
    config: dict,
    docker_image: Optional[str],
    reporter,
    force: bool = False,
    overwrite_existing: bool = False,
) -> None:
    project_config = config["project_config"]
    strategy_name = project_config["strategy_name"]
    recipe_name = project_config["model_to_train"]
    database_name = project_config["database_name"]
    database_session = project_config["database_session"]

    general_config = config["general_script_config"]

    # Snapshot tomado ANTES de que create_dataset/create_dataloader muten
    # config["torch_dataset"]/config["torch_dataloader"] in-place con
    # objetos vivos (X, y, dataset, generator -- mismo mutate-in-place que
    # repo_viejo, preservado verbatim más abajo). El trío de
    # reproducibilidad y mlflow.log_params necesitan el config tal como
    # vino del JSON, no la versión contaminada con tensores/Generators
    # (no serializables) que queda una vez que se procesó el primer
    # sujeto.
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
        extract_identity=_training_identity,
        extra_cleanup=lambda: [p.unlink() for p in output_path.rglob("*.csv")],
        context_label=f"la corrida de training en {output_path}",
    )
    # Trío de reproducibilidad a nivel barrido (identidad para
    # RunLifecycleManager) -- distinto del trío por-réplica que exige
    # PROTOCOL.md sección 9, escrito más abajo en cada carpeta hoja.
    log_reproducibility_trio(output_path, config_snapshot, docker_image)

    dataset_cfg = config["databases"]
    dataset = import_class(dataset_cfg["class_name"], dataset_cfg["module_name"])(**dataset_cfg["params"])

    sessions = getattr(dataset, "sessions", [])
    if database_session not in sessions:
        raise ValueError(f"Session {database_session} not found in dataset. Available: {sessions}")

    subjects_id = general_config.get("subjects_to_test") or dataset.subject_list
    model_init_seeds = general_config.get("model_init_seed") or [general_config["seed"]]

    partitions = [f"subject_{s:02d}" for s in subjects_id]
    partition_to_subject = {f"subject_{s:02d}": s for s in subjects_id}

    def _still_present(partition: str) -> bool:
        # 'success' para una partition (sujeto) implica que TODAS sus
        # réplicas (una por model_init_seed) llegaron a escribir
        # metrics_results.csv -- script_progress.csv tiene grano de
        # sujeto (PROTOCOL.md sección 6), no de réplica.
        subject_dir = output_path / partition
        if not subject_dir.exists():
            return False
        return all(
            (subject_dir / f"seed_{seed}" / "metrics_results.csv").exists()
            for seed in model_init_seeds
        )

    pending = lifecycle.prepare_partitions(partitions, still_present=_still_present)

    if not pending:
        logger.info("Todo ya procesado (%d particiones) -- nada para hacer.", len(partitions))
        return

    device = obtain_cuda_device()
    g = torch.Generator()
    g.manual_seed(general_config["seed"])

    splitter = WithinSubjectHoldoutSplitter(
        ptrain=general_config["ptrain"], pval=general_config["pval"], ptest=general_config["ptest"],
    )

    reporter.on_start(len(pending), meta={
        "strategy_name": strategy_name, "recipe_name": recipe_name, "database_name": database_name,
        "n_partitions": len(pending), "n_replicates_per_partition": len(model_init_seeds),
    })

    mlflow.set_experiment(f"{strategy_name}/{recipe_name}/{database_name}")

    for partition in pending:
        subject_id = partition_to_subject[partition]
        ts_start = lifecycle.mark_start(partition)
        reporter.on_run_start(partition)
        partition_status = "success"
        try:
            X, y, metadata = dataset.flatten_subject_data(subject_id, session=database_session)
            # NOTA: repo_viejo convertía labels string -> int vía un
            # EVENTS_DICT_EXTENDED global que no existe en el repo nuevo
            # (era una constante de global_config.py, nunca migrada --
            # ver PROTOCOL.md). El Splitter recibe y sin zar --
            # stratified_sequential_split (np.unique) no requiere labels
            # numéricos ni específicamente 0/1. ze_y se aplica en
            # materialize_fold, como último paso antes de tener un Fold con
            # datos reales (ver src/training/splitters.py).
            specs = splitter.split(X, y, metadata, subject_id=subject_id)

            for spec in specs:
                fold_partition = partition_name(spec)
                fold = materialize_fold(dataset, spec, default_session=database_session)

                torch.manual_seed(general_config["seed"])
                X_train = torch.tensor(fold.X_train, dtype=torch.float).to(device)
                X_val = torch.tensor(fold.X_val, dtype=torch.float).to(device)
                X_test = torch.tensor(fold.X_test, dtype=torch.float).to(device)
                y_train = torch.tensor(fold.y_train, dtype=torch.long).to(device)
                y_val = torch.tensor(fold.y_val, dtype=torch.long).to(device)
                y_test = torch.tensor(fold.y_test, dtype=torch.long).to(device)

                train_dataset = create_dataset('train', X_train, y_train, config['torch_dataset'])
                val_dataset = create_dataset('val', X_val, y_val, config['torch_dataset'])
                test_dataset = create_dataset('test', X_test, y_test, config['torch_dataset'])

                train_loader = create_dataloader('train', train_dataset, config['torch_dataloader'], g)
                val_loader = create_dataloader('val', val_dataset, config['torch_dataloader'], g)
                test_loader = create_dataloader('test', test_dataset, config['torch_dataloader'], g)

                for model_init_seed in model_init_seeds:
                    replicate = f"seed_{model_init_seed}"
                    run_dir = make_run_dir(strategy_name, recipe_name, database_name, fold_partition, replicate)
                    try:
                        set_seed(model_init_seed)

                        model_cfg = config["model"]
                        model = import_class(model_cfg["class_name"], model_cfg["module_name"])(**model_cfg["params"])
                        model.to(device)

                        criterion_cfg = config["criterion"]
                        criterion = import_class(criterion_cfg["class_name"], criterion_cfg["module_name"])(**criterion_cfg["params"])

                        optimizer_cfg = config["optimizer"]
                        optimizer = import_class(optimizer_cfg["class_name"], optimizer_cfg["module_name"])(
                            model.parameters(), **optimizer_cfg["params"]
                        )

                        base_callbacks = [
                            TimerCallback(),
                            LossTracker(name='train_loss', on_training=True),
                            LossTracker(name='val_loss', on_training=False),
                            EpochScoring(score_fn=accuracy, name='train_accuracy', on_training=True),
                            EpochScoring(score_fn=accuracy, name='val_accuracy', on_training=False),
                        ]
                        callbacks = build_callbacks(
                            base_callbacks=base_callbacks,
                            config_callbacks=config.get("callbacks"),
                            aditional_params={"path_to_save": str(run_dir)},
                        )
                        callbacks.append(MLflowCallback())

                        trainer = Trainer_DL(
                            model=model, loss_fn=criterion, optimizer=optimizer,
                            train_loader=train_loader, val_loader=val_loader,
                            max_epochs=general_config["max_epochs"], callbacks=callbacks,
                        )

                        tags = {
                            "project_name": project_config.get("project_name") or "",
                            "strategy_name": strategy_name,
                            "database_name": database_name,
                            "database_session": database_session,
                            "label": project_config.get("label") or "",
                            "subject_id": str(subject_id),
                            "seed": str(model_init_seed),
                        }
                        with mlflow.start_run(run_name=f"{fold_partition}/{replicate}", tags=tags) as run:
                            mlflow.log_params(_flatten_for_mlflow(config_snapshot))

                            trainer.train()

                            log_reproducibility_trio(run_dir, config_snapshot, docker_image)

                            history = trainer.get_history()
                            _write_train_curve_csv(run_dir / "train_curve.csv", history)

                            # Loader de evaluación final: copia del sub-config
                            # TOMADA DEL SNAPSHOT (config["torch_dataloader"] ya
                            # está mutado in-place con el Generator/Dataset vivos
                            # de este sujeto -- create_dataloader los inyecta ahí
                            # antes del loop de réplicas -- y esos objetos no son
                            # picklables/deepcopy-ables). Nunca mutando el config
                            # compartido (fix ya aprobado -- ver docstring del
                            # módulo).
                            eval_dl_config = copy.deepcopy(config_snapshot["torch_dataloader"])
                            eval_dl_config["train"]["params"]["shuffle"] = False
                            eval_train_loader = create_dataloader("train", train_dataset, eval_dl_config, g)

                            proba_train = trainer.infer(eval_train_loader, probability=True).cpu().numpy()
                            proba_val = trainer.infer(val_loader, probability=True).cpu().numpy()
                            proba_test = trainer.infer(test_loader, probability=True).cpu().numpy()

                            metrics_rows = _compute_final_metrics(
                                y_train, y_val, y_test, proba_train, proba_val, proba_test,
                            )
                            metrics_path = run_dir / "metrics_results.csv"
                            for row in metrics_rows:
                                _append_csv_row(metrics_path, row)
                                mlflow.log_metric(f"{row['split']}_{row['metric_name']}", row["value"])

                            (run_dir / ".mlflow_run_id").write_text(run.info.run_id)

                        reporter.on_run_result(f"{fold_partition}/{replicate}", "success")
                    except Exception as e:
                        logger.error("Fallo entrenando %s/%s: %s", fold_partition, replicate, e)
                        partition_status = "failed"
                        reporter.on_run_result(f"{fold_partition}/{replicate}", "failed")

        except Exception as e:
            logger.error("Fallo cargando/particionando %s: %s", partition, e)
            partition_status = "failed"

        lifecycle.mark_result(partition, partition_status, ts_start)

    logger.info("--- Training Finished ---")
    reporter.on_finish({"total": len(pending)})
