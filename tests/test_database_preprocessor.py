"""
Regresión: tras adoptar RunLifecycleManager, EEGDatabasePreprocessor debe
seguir produciendo el mismo run_registry.csv (mismas columnas, mismo
comportamiento de resume) que antes del refactor -- ver diff documentado en
src/utils/run_lifecycle.py y el plan de refactor.
"""
import numpy as np
import pandas as pd
import pytest

from src.preprocessing.database_preprocessor import EEGDatabasePreprocessor
from src.utils.paths import PATHS


class _FakeSubject:
    def __init__(self, subject_id, n_runs=1):
        self.subject_id = subject_id
        self._n_runs = n_runs

    def iter_all_runs(self):
        for i in range(self._n_runs):
            data = {"data": np.zeros((2, 3)), "labels": np.array([0, 1])}
            yield ("motor_imagery", f"run_{i+1}", data)


class _FakeDataset:
    code = "FakeDS"
    sessions = ["session_1"]
    data_to_load = ["motor_imagery"]

    def __init__(self, n_subjects=2):
        self._subjects = list(range(1, n_subjects + 1))

    def get_subjects_list(self):
        return self._subjects

    def get_data_available(self):
        return self.data_to_load

    def get_subject(self, subject, session=None):
        return _FakeSubject(subject)


def _config():
    return {"preprocessing_pipeline": {"stages": []}}


@pytest.fixture
def preproc_env(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    data_root.mkdir()
    monkeypatch.setattr(PATHS, "janus_data_root", data_root)
    return data_root


def test_run_registry_columns_match_protocol_contract(preproc_env):
    dataset = _FakeDataset(n_subjects=1)
    orchestrator = EEGDatabasePreprocessor(dataset, "test-preproc", _config())
    orchestrator.run()

    registry = pd.read_csv(orchestrator.output_path / "run_registry.csv")
    assert list(registry.columns) == [
        "partition", "subject_id", "session", "context", "run_id",
        "status", "timestamp_start", "timestamp_end", "output_path",
    ]
    assert set(registry["status"]) == {"success"}


def test_resume_skips_successful_partitions(preproc_env):
    dataset = _FakeDataset(n_subjects=2)
    orchestrator = EEGDatabasePreprocessor(dataset, "test-preproc", _config())
    orchestrator.run()

    registry_before = pd.read_csv(orchestrator.output_path / "run_registry.csv")
    assert len(registry_before) == 2

    orchestrator2 = EEGDatabasePreprocessor(dataset, "test-preproc", _config())
    orchestrator2.run()

    registry_after = pd.read_csv(orchestrator2.output_path / "run_registry.csv")
    assert len(registry_after) == 2  # nada nuevo agregado


def test_force_reprocesses_everything(preproc_env):
    dataset = _FakeDataset(n_subjects=1)
    orchestrator = EEGDatabasePreprocessor(dataset, "test-preproc", _config())
    orchestrator.run()

    orchestrator2 = EEGDatabasePreprocessor(dataset, "test-preproc", _config(), force=True)
    orchestrator2.run()

    registry = pd.read_csv(orchestrator2.output_path / "run_registry.csv")
    assert len(registry) == 2  # una fila por corrida, --force no filtra por status


def test_identity_mismatch_without_tty_rejects(preproc_env, monkeypatch):
    dataset = _FakeDataset(n_subjects=1)
    orchestrator = EEGDatabasePreprocessor(dataset, "test-preproc", _config())
    orchestrator.run()

    different_config = {"preprocessing_pipeline": {"stages": [{"stage_name": "new_stage", "stage_type": "steps", "steps": []}]}}
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    orchestrator2 = EEGDatabasePreprocessor(dataset, "test-preproc", different_config)
    with pytest.raises(RuntimeError):
        orchestrator2.run()


def test_overwrite_existing_resets_registry(preproc_env):
    dataset = _FakeDataset(n_subjects=1)
    orchestrator = EEGDatabasePreprocessor(dataset, "test-preproc", _config())
    orchestrator.run()

    different_config = {"preprocessing_pipeline": {"stages": [{"stage_name": "new_stage", "stage_type": "steps", "steps": []}]}}
    orchestrator2 = EEGDatabasePreprocessor(dataset, "test-preproc", different_config, overwrite_existing=True)
    orchestrator2.run()

    registry = pd.read_csv(orchestrator2.output_path / "run_registry.csv")
    assert len(registry) == 1  # reiniciado, no acumulado con la corrida anterior
