import pandas as pd
import pytest

mlflow = pytest.importorskip("mlflow")
torch = pytest.importorskip("torch")

from src.preprocessing.progress_reporter import ProgressReporter
from src.utils.paths import PATHS


def _base_config(max_epochs=2, model_init_seed=(1, 2), n_subjects=2):
    return {
        "script_type": "train_dl",
        "project_config": {
            "project_name": "TEST", "strategy_name": "WS-Standard",
            "model_to_train": "TinyClassifier", "database_name": "FakeDataset",
            "database_session": "session_1", "label": "smoke",
        },
        "databases": {
            "class_name": "FakeDataset", "module_name": "tests.fixtures.fake_bci",
            "params": {"n_subjects": n_subjects, "n_trials": 40, "n_channels": 4, "n_times": 16},
        },
        "general_script_config": {
            "subjects_to_test": None, "max_epochs": max_epochs,
            "model_init_seed": list(model_init_seed),
            "ptrain": 0.6, "pval": 0.2, "ptest": 0.2,
            "device": "cpu", "seed": 8, "sets_seeds": None,
        },
        "torch_dataset": {
            "class_name": "MIBCI_SimpleDataset", "module_name": "src.torch_utils",
            "params": {"classification": True},
        },
        "torch_dataloader": {
            "train": {"class_name": "DataLoader", "module_name": "torch.utils.data",
                       "params": {"batch_size": 8, "shuffle": True}},
            "val": {"class_name": "DataLoader", "module_name": "torch.utils.data",
                    "params": {"batch_size": 8, "shuffle": False}},
            "test": {"class_name": "DataLoader", "module_name": "torch.utils.data",
                     "params": {"batch_size": 8, "shuffle": False}},
        },
        "model": {
            "class_name": "TinyClassifier", "module_name": "tests.fixtures.fake_bci",
            "params": {"n_channels": 4, "n_times": 16, "n_classes": 2},
        },
        "optimizer": {"class_name": "Adam", "module_name": "torch.optim", "params": {"lr": 0.01}},
        "criterion": {"class_name": "CrossEntropyLoss", "module_name": "torch.nn", "params": {}},
        "callbacks": [],
    }


@pytest.fixture
def training_env(tmp_path, monkeypatch):
    results_root = tmp_path / "results"
    results_root.mkdir()
    monkeypatch.setattr(PATHS, "janus_results_root", results_root)
    mlflow.set_tracking_uri(f"file://{tmp_path / 'mlruns'}")
    return results_root


def test_training_sweep_end_to_end(training_env):
    from src.training.orchestrator import run_training_sweep

    config = _base_config()
    run_training_sweep(config, docker_image=None, reporter=ProgressReporter(), force=False, overwrite_existing=False)

    output_path = training_env / "WS-Standard" / "TinyClassifier" / "FakeDataset"
    progress = pd.read_csv(output_path / "script_progress.csv")
    assert list(progress.columns) == ["partition", "status", "timestamp_start", "timestamp_end"]
    assert set(progress["partition"]) == {"subject_01", "subject_02"}
    assert set(progress["status"]) == {"success"}

    for subject in ("subject_01", "subject_02"):
        for seed in (1, 2):
            leaf = output_path / subject / f"seed_{seed}"
            assert (leaf / "config.json").exists()
            assert (leaf / ".git_commit").exists()
            assert (leaf / "train_curve.csv").exists()
            assert (leaf / ".mlflow_run_id").exists()

            metrics = pd.read_csv(leaf / "metrics_results.csv")
            assert list(metrics.columns) == ["split", "metric_name", "value"]
            assert set(metrics["split"]) == {"train", "val", "test"}
            assert set(metrics["metric_name"]) >= {"accuracy", "auc", "recall_0", "recall_1", "precision_0", "precision_1"}


def test_training_sweep_resume_skips_done_subjects(training_env):
    from src.training.orchestrator import run_training_sweep

    # Cada llamada usa su PROPIO dict (fresco), igual que en producción real
    # -- scripts/run_production.py parsea el JSON de cero en cada invocación
    # de proceso; create_dataset/create_dataloader mutan ese dict in-place
    # (verbatim repo_viejo), así que reusar el mismo objeto entre dos
    # llamadas dentro de un mismo test no refleja un escenario real.
    run_training_sweep(_base_config(max_epochs=1, model_init_seed=(1,), n_subjects=2),
                        docker_image=None, reporter=ProgressReporter())

    output_path = training_env / "WS-Standard" / "TinyClassifier" / "FakeDataset"
    progress = pd.read_csv(output_path / "script_progress.csv")
    assert len(progress) == 2

    # Segunda corrida con un config de igual contenido (identidad
    # coincide): ambos sujetos ya 'success' con su metrics_results.csv por
    # réplica todavía en disco -> nada para hacer, no se agregan filas.
    run_training_sweep(_base_config(max_epochs=1, model_init_seed=(1,), n_subjects=2),
                        docker_image=None, reporter=ProgressReporter())
    progress_after = pd.read_csv(output_path / "script_progress.csv")
    assert len(progress_after) == 2


def test_training_sweep_rejects_identity_mismatch_without_tty(training_env, monkeypatch):
    from src.training.orchestrator import run_training_sweep

    config = _base_config(max_epochs=1, model_init_seed=(1,), n_subjects=1)
    run_training_sweep(config, docker_image=None, reporter=ProgressReporter())

    changed = _base_config(max_epochs=1, model_init_seed=(1,), n_subjects=1)
    changed["general_script_config"]["ptest"] = 0.3  # part of identity -> mismatch
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    with pytest.raises(RuntimeError):
        run_training_sweep(changed, docker_image=None, reporter=ProgressReporter())
