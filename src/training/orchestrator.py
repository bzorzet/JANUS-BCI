"""
TrainingOrchestrator -- clase que coordina un barrido completo de
entrenamiento: RunLifecycleManager, trío de reproducibilidad, MLflow, y
el loop de unidades -> Schema.generate_folds() -> Trainer_DL/Trainer_ML.
 
Completamente agnóstico de qué Schema se usa: siempre llama
schema.units_to_process() para saber qué recorrer, y
schema.generate_folds(unit_id=...) con la misma keyword unificada para
los 3 Schemas (WithinUnitHoldoutSchema, WithinUnitKFoldSchema,
LeaveOneUnitOutSchema) -- nunca pregunta "¿qué Schema es este?".
 
El Generator de DataLoaders se arma DESDE fold.metadata["dataloader_seed"],
nunca creado una sola vez para todo el barrido -- cada Schema decide su
propia semántica de reseteo (fijo para Holdout/LOSO, derivado por
fold_idx para KFold), el orquestador solo usa lo que recibe.
 
Nota: CallbackDispatcher NO se instancia acá -- Trainer_DL ya lo compone
internamente en su propio __init__ (ver training/core/trainer.py). El
orquestador solo arma la LISTA de callbacks (vía build_callbacks) y se la
pasa a Trainer_DL, que es quien envuelve esa lista en su dispatcher.
Instanciar CallbackDispatcher acá también sería duplicar esa
responsabilidad -- import muerto corregido.
 
Config esperado (ver sesión de diseño):
  project_config: {project_name, strategy_name, model_to_train,
                    database_name, database_session, label}
  script_type: "train_dl" | "train_ml"
  databases: {class_name, module_name, params}       -- el dataset real
  data_provider: {class_name, module_name, params}   -- OPCIONAL, default BCIDataProviderAdapter
  schema: {class_name, module_name, params: {..., units_to_test, dataloader_seed, ...}}
  orchestrator: {model_init_seed, device}             -- sin subjects_to_test/ptrain/etc (viven en schema.params)
  model, optimizer, criterion, callbacks: como ya existían
  torch_dataset, torch_dataloader: {train, train_eval, val, test}  -- train_eval nuevo, mismos datos que train con shuffle=false
  trainer: {params: {max_epochs, ...}}                -- max_epochs migró acá desde general_script_config
  metrics_strategy: {class_name, module_name, params} -- OPCIONAL, default ClassificationMetrics()
"""
import logging
from pathlib import Path
from typing import List, Optional
import mlflow
import numpy as np
import torch

from src.training.data_provider import DataProvider, BCIDataProviderAdapter
from src.training.core.trainer import Trainer_DL, Trainer_ML
from src.training.core.seeds import set_seed
from src.training.tracking.callbacks_mlflow import MLflowCallback
from src.training.splitting.schemas import SCHEMA_REGISTRY
from src.training.metrics.classification import ClassificationMetrics
from src.training.utils import create_dataset, create_dataloader, build_callbacks
from src.utils.imports import import_class
from src.utils.json_utils import save_dict_as_json
from src.utils.paths import PATHS
from src.utils.reproducibility import log_reproducibility_trio
from src.utils.run_lifecycle import RunLifecycleManager, _append_csv_row

logger = logging.getLogger(__name__)

TRAINER_REGISTRY = {
    "train_dl": Trainer_DL,
    "train_ml": Trainer_ML,
}


class TrainingOrchestrator:
    """Coordina un barrido completo de training. Instanciar una vez por
    corrida (un config), llamar .run()."""

    def __init__(
        self, config: dict, docker_image: Optional[str] = None,
        reporter=None, force: bool = False, overwrite_existing: bool = False,
    ):
        self.config = config
        self.docker_image = docker_image
        self.reporter = reporter
        self.force = force
        self.overwrite_existing = overwrite_existing

        project_config = config["project_config"]
        self.strategy_name = project_config["strategy_name"]
        self.recipe_name = project_config["model_to_train"]
        self.database_name = project_config["database_name"]
        self.database_session = project_config["database_session"]
        self.script_type = config.get("script_type", "train_dl")

        self.output_path = PATHS.results_root / self.strategy_name / self.recipe_name / self.database_name

        self.provider: Optional[DataProvider] = None
        self.schema = None
        self.lifecycle: Optional[RunLifecycleManager] = None
        self.metrics_strategy = None

    # --- setup, cada uno una responsabilidad chica y nombrada ---

    def _setup_provider(self) -> DataProvider:
        """Instancia el dataset real vía import_class (config['databases']),
        lo envuelve en el adapter configurado en config['data_provider']
        (o BCIDataProviderAdapter por default si no se especifica)."""
        dataset_cfg = self.config["databases"]
        dataset = import_class(dataset_cfg["class_name"], dataset_cfg["module_name"])(**dataset_cfg["params"])

        provider_cfg = self.config.get("data_provider")
        if provider_cfg is None:
            return BCIDataProviderAdapter(dataset, default_session=self.database_session)

        provider_class = import_class(provider_cfg["class_name"], provider_cfg["module_name"])
        params = dict(provider_cfg.get("params", {}))
        params["dataset"] = dataset
        return provider_class(**params)

    def _setup_schema(self, provider: DataProvider):
        """Instancia el Schema vía import_class desde config['schema'].
        split_strategy y label_transform, si están en el config, también
        se resuelven vía import_class y se inyectan -- si no están, el
        Schema usa sus propios defaults (ver schemas.py). units_to_test y
        dataloader_seed, si están, van directo en schema['params'] --
        no hay tratamiento especial, son params del Schema como cualquier
        otro."""
        schema_cfg = self.config["schema"]
        schema_class = SCHEMA_REGISTRY.get(schema_cfg["class_name"])
        if schema_class is None:
            schema_class = import_class(schema_cfg["class_name"], schema_cfg["module_name"])

        params = dict(schema_cfg.get("params", {}))
        params["provider"] = provider

        if "split_strategy" in schema_cfg:
            ss_cfg = schema_cfg["split_strategy"]
            params["split_strategy"] = import_class(ss_cfg["class_name"], ss_cfg["module_name"])(**ss_cfg.get("params", {}))
        if "label_transform" in schema_cfg:
            lt_cfg = schema_cfg["label_transform"]
            params["label_transform"] = import_class(lt_cfg["class_name"], lt_cfg["module_name"])(**lt_cfg.get("params", {}))

        return schema_class(**params)

    def _setup_metrics_strategy(self):
        """Resuelve MetricsStrategy desde config['metrics_strategy'], o
        ClassificationMetrics() default (solo accuracy) si no está."""
        metrics_cfg = self.config.get("metrics_strategy")
        if metrics_cfg is None:
            return ClassificationMetrics()
        metrics_class = import_class(metrics_cfg["class_name"], metrics_cfg["module_name"])
        return metrics_class(**metrics_cfg.get("params", {}))

    @staticmethod
    def _resolve_model_init_seeds(orchestrator_config: dict) -> List[int]:
        """Forma A: lista fija explícita. Forma B: {n_seeds, seed} ->
        genera n_seeds enteros aleatorios, reproducible vía seed maestro."""
        spec = orchestrator_config["model_init_seed"]
        if isinstance(spec, list):
            return spec
        if isinstance(spec, dict):
            rng = np.random.RandomState(spec["seed"])
            return [int(s) for s in rng.randint(0, 1_000_000, size=spec["n_seeds"])]
        raise ValueError(f"model_init_seed debe ser una lista o un dict {{n_seeds, seed}}, es {type(spec)}")

    # --- ejecución ---

    def run(self) -> None:
        self.provider = self._setup_provider()
        self.schema = self._setup_schema(self.provider)
        self.metrics_strategy = self._setup_metrics_strategy()

        self.lifecycle = RunLifecycleManager(
            self.output_path,
            force=self.force, overwrite_existing=self.overwrite_existing,
            emit_running_row=True,
        )
        self.lifecycle.check_identity_or_raise(
            new_config=self.config,
            context_label=f"la corrida de training en {self.output_path}",
        )
        log_reproducibility_trio(self.output_path, self.config, self.docker_image)

        # El orquestador nunca sabe qué es una "unidad" ni cómo se filtra
        # -- le pregunta al Schema, que tiene su propio vocabulario
        # (units_to_test) resuelto en su constructor.
        units = self.schema.units_to_process()

        orchestrator_cfg = self.config["orchestrator"]
        model_init_seeds = self._resolve_model_init_seeds(orchestrator_cfg)

        def _still_present(partition: str) -> bool:
            return (self.output_path / partition).exists()

        pending_units = self.lifecycle.prepare_partitions(units, still_present=_still_present)

        if not pending_units:
            logger.info("Todo ya procesado (%d unidades) -- nada para hacer.", len(units))
            return

        device = torch.device(orchestrator_cfg.get("device", "cuda") if torch.cuda.is_available() else "cpu")
        mlflow.set_experiment(f"{self.strategy_name}/{self.recipe_name}/{self.database_name}")

        if self.reporter:
            self.reporter.on_start(len(pending_units), meta={
                "strategy_name": self.strategy_name, "recipe_name": self.recipe_name,
                "database_name": self.database_name, "n_units": len(pending_units),
            })

        for unit_id in pending_units:
            self._run_unit(unit_id, model_init_seeds, device)

        logger.info("--- Training Finished ---")
        if self.reporter:
            self.reporter.on_finish({"total": len(pending_units)})

    def _run_unit(self, unit_id, model_init_seeds: List[int], device) -> None:
        """Una unidad puede producir MÁS DE UN Fold (ej. WithinUnitKFoldSchema
        con K folds) -- cada Fold es una partición independiente en disco,
        cada una recorre las N seeds de init de pesos."""
        ts_start = self.lifecycle.mark_start(str(unit_id))
        status = "success"

        try:
            folds = self.schema.generate_folds(unit_id=unit_id)  # misma keyword para los 3 Schemas

            for fold in folds:
                partition = self.schema.partition_name(fold)
                run_dir = self.output_path / partition
                run_dir.mkdir(parents=True, exist_ok=True)
                save_dict_as_json(str(run_dir), fold.metadata, "fold_metadata.json")

                if self.reporter:
                    self.reporter.on_run_start(partition)

                # Generator armado desde lo que EL SCHEMA decidió, no
                # creado una sola vez para todo el barrido -- fix del
                # hallazgo del Generator compartido.
                g = torch.Generator()
                g.manual_seed(fold.metadata["dataloader_seed"])

                for seed in model_init_seeds:
                    self._run_replicate(fold, partition, seed, g, device)

                if self.reporter:
                    self.reporter.on_run_result(partition, "success")

        except Exception as e:
            logger.error("Fallo procesando unidad %s: %s", unit_id, e)
            status = "failed"
            if self.reporter:
                self.reporter.on_run_result(str(unit_id), "failed")

        self.lifecycle.mark_result(str(unit_id), status, ts_start)

    def _run_replicate(self, fold, partition: str, seed: int, g: torch.Generator, device) -> None:
        """Una réplica = un (fold, seed). set_seed afecta SOLO init de
        pesos -- g (shuffling de DataLoaders) es compartido entre las N
        seeds del MISMO fold, a propósito (mismo Generator, mismo estado
        de shuffling potencial, para que la única variable entre réplicas
        sea la seed de init de pesos)."""
        set_seed(seed)

        run_dir = self.output_path / partition  # guardado aplanado -- detalle completo en Paso 4b

        model_cfg = self.config["model"]
        model = import_class(model_cfg["class_name"], model_cfg["module_name"])(**model_cfg["params"]).to(device)

        trainer_class = TRAINER_REGISTRY[self.script_type]

        tags = {
            "project_name": self.config["project_config"].get("project_name", ""),
            "strategy_name": self.strategy_name,
            "database_name": self.database_name,
            "label": self.config["project_config"].get("label", ""),
            "partition": partition,
            "model_init_seed": str(seed),
        }

        with mlflow.start_run(run_name=f"{partition}_seed{seed}", tags=tags):
            if trainer_class is Trainer_DL:
                self._train_dl_replicate(fold, model, run_dir, seed, g, device)
            else:
                self._train_ml_replicate(fold, model, run_dir, seed)

    def _write_outputs_csv(fold, y_pred_train, y_pred_val, y_pred_test, run_dir, seed):
        """Un CSV único por seed: trial_idx (absoluto, mismo espacio que
        fold.metadata['train_idx']/etc.), split, y_true, y_pred,
        proba_0..proba_N. Reemplaza train/val/test_outputs.csv separados."""
        import pandas as pd

        rows = []
        splits = [
            ("train", fold.metadata["train_idx"], fold.y_train, y_pred_train),
            ("test", fold.metadata["test_idx"], fold.y_test, y_pred_test),
        ]
        if fold.X_val is not None:
            splits.insert(1, ("val", fold.metadata["val_idx"], fold.y_val, y_pred_val))

        for split_name, idx_list, y_true, y_pred_proba in splits:
            y_pred_class = y_pred_proba.argmax(axis=1)  # asume y_pred_proba viene como (n_trials, n_clases)
            n_classes = y_pred_proba.shape[1]
            for i, trial_idx in enumerate(idx_list):
                row = {
                    "trial_idx": trial_idx, "split": split_name,
                    "y_true": int(y_true[i]), "y_pred": int(y_pred_class[i]),
                }
                for cls in range(n_classes):
                    row[f"proba_{cls}"] = float(y_pred_proba[i, cls])
                rows.append(row)

        outputs_dir = run_dir / "outputs"
        outputs_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(outputs_dir / f"seed_{seed}_outputs.csv", index=False)

    def _train_dl_replicate(self, fold, model, run_dir, seed, g: torch.Generator, device) -> None:
        optimizer_cfg = self.config["optimizer"]
        optimizer = import_class(optimizer_cfg["class_name"], optimizer_cfg["module_name"])(model.parameters(), **optimizer_cfg["params"])

        criterion_cfg = self.config["criterion"]
        criterion = import_class(criterion_cfg["class_name"], criterion_cfg["module_name"])(**criterion_cfg["params"])

        train_dataset = create_dataset("train", fold.X_train, fold.y_train, self.config["torch_dataset"])
        train_loader = create_dataloader("train", train_dataset, self.config["torch_dataloader"], g)

        val_loader = None
        if fold.X_val is not None:
            val_dataset = create_dataset("val", fold.X_val, fold.y_val, self.config["torch_dataset"])
            val_loader = create_dataloader("val", val_dataset, self.config["torch_dataloader"], g)

        trainer_cfg = self.config.get("trainer", {})
        weights_dir = run_dir / "weights"  # única declaración, reusada abajo
        callbacks = build_callbacks(
            base_callbacks=[MLflowCallback()],
            config_callbacks=self.config.get("callbacks"),
            aditional_params={
                "path_to_save": str(weights_dir),
                "prefix": f"seed_{seed}_",
                "optimizer": optimizer,
            },
        )

        trainer = Trainer_DL(
            model=model, loss_fn=criterion, optimizer=optimizer,
            train_loader=train_loader, val_loader=val_loader,
            max_epochs=trainer_cfg.get("params", {}).get("max_epochs", 100),
            callbacks=callbacks,
        )
        trainer.train()

        # --- Inferencia explícita post-training, órdenes preservados ---
        # train_eval: mismos datos que train, pero shuffle=False -- distinto
        # PROPÓSITO de train (entrenar vs. registrar predicciones ordenadas),
        # mismo patrón que ya usaba el config para val/test.
        train_eval_dataset = create_dataset("train_eval", fold.X_train, fold.y_train, self.config["torch_dataset"])
        train_eval_loader = create_dataloader("train_eval", train_eval_dataset, self.config["torch_dataloader"], g=None)
        y_pred_train = trainer.infer(dataloader=train_eval_loader, probability=True).cpu().numpy()

        y_pred_val = None
        if val_loader is not None:
            val_eval_dataset = create_dataset("val", fold.X_val, fold.y_val, self.config["torch_dataset"])
            val_eval_loader = create_dataloader("val", val_eval_dataset, self.config["torch_dataloader"], g=None)
            y_pred_val = trainer.infer(dataloader=val_eval_loader, probability=True).cpu().numpy()

        test_dataset = create_dataset("test", fold.X_test, fold.y_test, self.config["torch_dataset"])
        test_loader = create_dataloader("test", test_dataset, self.config["torch_dataloader"], g=None)
        y_pred_test = trainer.infer(dataloader=test_loader, probability=True).cpu().numpy()

        self._write_outputs_csv(fold, y_pred_train, y_pred_val, y_pred_test, run_dir, seed)

        # --- Métricas de test vía MetricsStrategy ---
        y_test_pred_class = y_pred_test.argmax(axis=1)
        metrics_rows = self.metrics_strategy.compute(fold.y_test, y_test_pred_class, y_pred_proba=y_pred_test, split="test")
        for row in metrics_rows:
            row["partition"] = run_dir.name
            row["model_init_seed"] = seed
            _append_csv_row(self.output_path / "metrics_results.csv", row)
            mlflow.log_metric(row["metric_name"], row["value"])

    def _train_ml_replicate(self, fold, model, run_dir: Path, seed: int) -> None:
        """Análogo a _train_dl_replicate pero para Trainer_ML -- sin
        DataLoaders, sin device. TODO: guardado de pesos (joblib, no
        .pth -- ver nota de Trainer_ML) y outputs CSV, pendiente de
        definir el mecanismo de checkpoint equivalente a CustomCheckpoint
        para modelos sklearn-like."""
        trainer = Trainer_ML(model, fold.X_train, fold.y_train, X_val=fold.X_val, y_val=fold.y_val)
        trainer.train()
 
        y_pred_train = trainer.infer(fold.X_train, probability=True)
        y_pred_val = trainer.infer(fold.X_val, probability=True) if fold.X_val is not None else None
        y_pred_test = trainer.infer(fold.X_test, probability=True)
 
        self._write_outputs_csv(fold, y_pred_train, y_pred_val, y_pred_test, run_dir, seed)
 
        y_test_pred_class = y_pred_test.argmax(axis=1)
        metrics_rows = self.metrics_strategy.compute(fold.y_test, y_test_pred_class, y_pred_proba=y_pred_test, split="test")
        for row in metrics_rows:
            row["partition"] = run_dir.name
            row["model_init_seed"] = seed
            _append_csv_row(self.output_path / "metrics_results.csv", row)
            mlflow.log_metric(row["metric_name"], row["value"])