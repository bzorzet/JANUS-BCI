"""
Orquestador único de preprocesamiento (preprocessing/DESIGN.md sección 4).
Reemplaza a `EEGDatabasePreprocessor`/`EEGDatabaseICAPreprocessor` de
repo_viejo — una forma nueva de preprocesar se agrega registrando un
stage handler (`EEGPreprocessor.register_stage_handler`) o un artifact
saver (`register_artifact_saver`), no subclaseando este orquestador.
"""
import csv
import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import mne
import numpy as np

from src.utils.json_utils import save_dict_as_json
from src.utils.paths import PATHS
from src.utils.reproducibility import log_reproducibility_trio

from .artifact_savers import ARTIFACT_SAVERS
from .preprocess_eeg import EEGPreprocessor, StageResult


def _save_processed_data(data: Any, base_path: Path, prefix: str) -> Optional[Path]:
    """Raw -> .fif, Epochs -> .epo.fif, dict -> .npy. Puerto verbatim de
    _save_results/_save_processed_data (idénticas en ambas clases viejas)."""
    base_path.mkdir(parents=True, exist_ok=True)
    if isinstance(data, mne.io.BaseRaw):
        out = base_path / f"{prefix}_raw.fif"
        data.save(out, overwrite=True, verbose=False)
        return out
    elif isinstance(data, mne.BaseEpochs):
        out = base_path / f"{prefix}_epo.fif"
        data.save(out, overwrite=True, verbose=False)
        return out
    elif isinstance(data, dict):
        last_out = None
        if "data" in data:
            data_content = data["data"]
            if isinstance(data_content, dict):
                for key, val in data_content.items():
                    if isinstance(val, np.ndarray):
                        last_out = base_path / f"{prefix}_{key}.npy"
                        np.save(last_out, val)
            elif isinstance(data_content, np.ndarray):
                last_out = base_path / f"{prefix}_data.npy"
                np.save(last_out, data_content)
        if "labels" in data and isinstance(data["labels"], np.ndarray) and len(data["labels"]) > 0:
            last_out = base_path / f"{prefix}_labels.npy"
            np.save(last_out, data["labels"])
        return last_out
    return None


def _append_csv_row(path: Path, row: Dict[str, Any]) -> None:
    """Append + flush por fila (abre/escribe/cierra) — nada se acumula en
    memoria, así un barrido que se cae a mitad de camino deja consultable
    todo lo ya procesado (DESIGN.md sección 4 punto 4)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def _now() -> str:
    return datetime.datetime.now().isoformat()


class EEGDatabasePreprocessor:
    def __init__(self, dataset, preprocessing_name: str, config: dict, docker_image: Optional[str] = None):
        self.dataset = dataset
        self.preprocessing_name = preprocessing_name
        self.config = config
        self.docker_image = docker_image
        self.output_path = PATHS.preprocessed_root / preprocessing_name / dataset.code
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.worker = EEGPreprocessor(stages=config["preprocessing_pipeline"]["stages"])
        self._artifact_savers: Dict[str, Callable] = dict(ARTIFACT_SAVERS)

    def register_artifact_saver(self, key: str, fn: Callable) -> None:
        self._artifact_savers[key] = fn

    def run(self) -> None:
        # Trío de reproducibilidad + manifest, una sola vez, antes del loop
        # (DESIGN.md sección 4 punto 5).
        log_reproducibility_trio(self.output_path, self.config, self.docker_image)
        self._write_dataset_description()

        subjects_list = self.dataset.get_subjects_list()
        sessions = getattr(self.dataset, "sessions", ["session_1"])
        print(f"Starting preprocessing for {len(subjects_list)} subjects...")

        for subject in subjects_list:
            for session in sessions:
                try:
                    subject_obj = self.dataset.get_subject(subject, session=session)
                except Exception as e:
                    print(f"  CRITICAL ERROR loading subject {subject}: {e}")
                    self._append_run_registry(
                        partition=f"subject_{subject:02d}/unknown/unknown",
                        subject=subject,
                        session=session,
                        context=None,
                        run_id=None,
                        status="failed",
                        ts_start=_now(),
                        ts_end=_now(),
                        output_path=None,
                    )
                    continue
                if subject_obj is None:
                    continue
                for context, run_id, raw_data in subject_obj.iter_all_runs():
                    self._process_one_run(subject, session, context, run_id, raw_data)

        print("--- Preprocessing Finished ---")

    def _process_one_run(self, subject, session, context, run_id, raw_data) -> None:
        partition = f"subject_{subject:02d}/{context}/{run_id}"
        prefix = f"subject_{subject:02d}_{context}_{run_id}"
        ts_start = _now()
        status = "failed"
        output_path = None
        try:
            print(f"Processing {partition}")
            result = self.worker.process(raw_data)
            data_dir = self.output_path / session / f"subject_{subject:02d}"
            output_path = _save_processed_data(result.data, data_dir, prefix)
            self._save_artifacts(result, subject, prefix)
            self._append_stage_metrics(result.metrics, partition)
            self._append_detail_tables(result.detail_tables, partition)
            status = "success"
        except Exception as e:
            print(f"  ERROR processing {partition}: {e}")
        finally:
            ts_end = _now()
            self._append_run_registry(
                partition=partition,
                subject=subject,
                session=session,
                context=context,
                run_id=run_id,
                status=status,
                ts_start=ts_start,
                ts_end=ts_end,
                output_path=str(output_path) if output_path else None,
            )

    def _save_artifacts(self, result: StageResult, subject, prefix: str) -> None:
        # Un saver que falla (ej. no puede graficar por falta de montage)
        # no debe tirar abajo el run entero ni perder metrics/detail_tables
        # ya calculados — mismo espíritu que "sin saver registrado, se
        # omite" (DESIGN.md sección 4 punto 3): nunca rompe el barrido.
        for stage_name, stage_artifacts in result.artifacts.items():
            stage_detail_tables = result.detail_tables.get(stage_name, {})
            for key, value in stage_artifacts.items():
                saver = self._artifact_savers.get(key)
                if saver is None:
                    print(f"  (info) sin saver registrado para artifact '{key}' del stage '{stage_name}', se omite")
                    continue
                output_dir = self.output_path / "artifacts" / stage_name / f"subject_{subject:02d}"
                try:
                    saver(
                        value,
                        artifacts=stage_artifacts,
                        detail_tables=stage_detail_tables,
                        output_dir=output_dir,
                        prefix=prefix,
                    )
                except Exception as e:
                    print(f"  (warn) saver de artifact '{key}' del stage '{stage_name}' falló: {e}")

    def _append_stage_metrics(self, metrics: Dict[str, dict], partition: str) -> None:
        path = self.output_path / "stage_metrics.csv"
        for stage_name, stage_metrics in metrics.items():
            for metric_name, value in stage_metrics.items():
                _append_csv_row(path, {
                    "partition": partition,
                    "stage_name": stage_name,
                    "metric_name": metric_name,
                    "value": value,
                })

    def _append_detail_tables(self, detail_tables: Dict[str, dict], partition: str) -> None:
        for stage_name, stage_details in detail_tables.items():
            for detail_key, rows in stage_details.items():
                path = self.output_path / f"{stage_name}_{detail_key}.csv"
                for row in rows:
                    _append_csv_row(path, {"partition": partition, **row})

    def _append_run_registry(
        self, *, partition, subject, session, context, run_id, status, ts_start, ts_end, output_path
    ) -> None:
        _append_csv_row(self.output_path / "run_registry.csv", {
            "partition": partition,
            "subject_id": subject,
            "session": session,
            "context": context,
            "run_id": run_id,
            "status": status,
            "timestamp_start": ts_start,
            "timestamp_end": ts_end,
            "output_path": output_path,
        })

    def _write_dataset_description(self) -> None:
        info = {
            "dataset_code": getattr(self.dataset, "code", "Unknown"),
            "subject_list": self.dataset.get_subjects_list(),
            "sessions_availables": getattr(self.dataset, "sessions", ["session_1"]),
            # Fix de la deriva DESIGN.md sección 1: siempre lo que
            # realmente se cargó (y por lo tanto quedó escrito a disco),
            # nunca la lista fija de contextos posibles del dataset.
            "data_availables": getattr(self.dataset, "data_to_load", None) or self.dataset.get_data_available(),
            "sfreq": getattr(self.dataset, "sfreq", None),
            "ch_names": getattr(self.dataset, "ch_names", []),
            "ch_types": getattr(self.dataset, "ch_types", []),
            "event_id": getattr(self.dataset, "event_id", {}),
            "standard_montage": getattr(self.dataset, "standard_montage", None),
            "subjects_individual_metadata": {},
        }
        subjects_metadata = getattr(self.dataset, "subjects_metadata", None)
        if subjects_metadata is not None:
            for _, row in subjects_metadata.iterrows():
                sub_id = row.get("subject_id", row.get("id", row.get("Subject")))
                if sub_id is not None:
                    info["subjects_individual_metadata"][int(sub_id)] = row.to_dict()
        save_dict_as_json(str(self.output_path), info, "dataset_description.json")
