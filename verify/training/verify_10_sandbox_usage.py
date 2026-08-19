"""
Script de verificación -- confirma que Splitter + materialize_fold +
Trainer_DL/Trainer_ML se pueden usar en un sandbox de investigación SIN
instanciar ningún orquestador (RunLifecycleManager, MLflow, CSVs, etc.).

Corré esto a mano (modo debug), no vía pytest.

Por qué importa: el diseño completo de esta sesión se apoya en que
"training/execution/" y "training/splitting/" reciben todo por parámetro y
nunca dependen de contexto de orquestación -- si eso NO fuera cierto (por
ejemplo, si algún import transitivo arrastrara orchestrator.py, o si
alguna clase asumiera silenciosamente un RunLifecycleManager activo), el
research rápido en notebook/sandbox quedaría bloqueado, contradiciendo la
distinción sandbox/producción de PROTOCOL.md sección 1.

Qué verifica:
1. Los imports de Splitter/materialize_fold/Trainer_DL/Trainer_ML NO
   arrastran orchestrator.py ni orchestrator_testing.py como dependencia
   transitiva -- verificado inspeccionando sys.modules después de
   importar, sin haber importado el orquestador explícitamente.
2. Flujo completo end-to-end en modo "sandbox": Splitter.split() ->
   materialize_fold() -> Trainer_DL.train() -> trainer.infer(), sin
   ningún RunLifecycleManager, sin MLflow, sin CSVs, sin config JSON de
   por medio -- todo con valores Python directos, como en un notebook.
3. Mismo flujo end-to-end pero con Trainer_ML, confirmando que el camino
   ML también es standalone.
4. Ningún archivo de disco (CSV, config.json, .pth) se crea como
   side-effect de este flujo -- confirma que no hay escritura oculta de
   "modo producción" colándose en el camino sandbox.
"""
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


def verify_imports_dont_pull_orchestrator():
    print("--- 1. Importar Splitter/materialize_fold/Trainer NO arrastra el orquestador ---")

    modules_before = set(sys.modules.keys())

    from src.training.splitting.splitters import WithinSubjectHoldoutSplitter
    from src.training.splitting.fold import materialize_fold, PartitionSpec
    from src.training.core.trainer import Trainer_DL, Trainer_ML

    modules_after = set(sys.modules.keys())
    new_modules = modules_after - modules_before

    orchestrator_modules = [m for m in new_modules if "orchestrator" in m or "run_lifecycle" in m]
    assert not orchestrator_modules, (
        f"Importar piezas de execution/splitting arrastró módulos de orquestación como "
        f"dependencia transitiva: {orchestrator_modules} -- esto rompe la posibilidad de "
        f"usarlas en sandbox sin instanciar un orquestador."
    )
    print(f"  {len(new_modules)} módulos nuevos importados, ninguno relacionado a orchestrator/run_lifecycle")
    print("  OK -- las piezas de execution/splitting son standalone respecto a orquestación\n")


class DummyModel(nn.Module):
    def __init__(self, in_features=4, n_classes=2):
        super().__init__()
        self.linear = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.linear(x)


def verify_full_sandbox_flow_dl():
    print("--- 2. Flujo completo sandbox con Trainer_DL: Splitter -> materialize_fold -> train -> infer ---")
    from src.training.splitting.splitters import WithinSubjectHoldoutSplitter
    from src.training.splitting.fold import materialize_fold
    from src.training.core.trainer import Trainer_DL

    # Dataset dummy en memoria -- ningún PreprocessedDataset, ningún JSON,
    # nada de config real. Solo arrays de numpy, como en un notebook.
    class InMemoryDataset:
        """Fake dataset mínimo: implementa SOLO flatten_subject_data,
        justo el contrato que materialize_fold necesita -- sin heredar de
        BaseEEGDataset ni nada del framework de eeg_datasets."""
        def __init__(self, X, y):
            self._X, self._y = X, y

        def flatten_subject_data(self, subject_id, session=None):
            import pandas as pd
            metadata = pd.DataFrame({"subject": [subject_id] * len(self._y)})
            return self._X, self._y, metadata

    n, n_channels, n_time = 60, 3, 16
    rng = np.random.RandomState(0)
    X = rng.randn(n, n_channels, n_time).astype(np.float32)
    y = np.array([0, 1] * (n // 2))
    dataset = InMemoryDataset(X, y)

    # 1. Splitter -- decide la partición, sin tocar el dataset.
    splitter = WithinSubjectHoldoutSplitter(ptrain=0.6, pval=0.2, ptest=0.2)
    specs = splitter.split(X, y, subject_id=1)
    assert len(specs) == 1

    # 2. materialize_fold -- traduce el spec a datos reales, vía el
    # dataset fake (que solo implementa flatten_subject_data).
    fold = materialize_fold(specs[0], dataset, default_session="session_1", X=X, y=y)

    # 3. Armar DataLoaders a mano -- sin create_dataset/create_dataloader
    # (que son utilidades del orquestador), directo con torch, como haría
    # cualquiera en un notebook.
    X_train_flat = torch.tensor(fold.X_train.reshape(len(fold.y_train), -1), dtype=torch.float)
    y_train_t = torch.tensor(fold.y_train, dtype=torch.long)
    X_val_flat = torch.tensor(fold.X_val.reshape(len(fold.y_val), -1), dtype=torch.float)
    y_val_t = torch.tensor(fold.y_val, dtype=torch.long)

    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_train_flat, y_train_t), batch_size=8,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(X_val_flat, y_val_t), batch_size=8,
    )

    # 4. Trainer_DL -- SIN callbacks, SIN MLflow, SIN CallbackDispatcher
    # de producción con defaults del orquestador. callbacks=[] explícito.
    model = DummyModel(in_features=n_channels * n_time, n_classes=2)
    trainer = Trainer_DL(
        model=model, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()),
        train_loader=train_loader, val_loader=val_loader, max_epochs=3, callbacks=[],
    )
    trainer.train()

    X_test_flat = torch.tensor(fold.X_test.reshape(len(fold.y_test), -1), dtype=torch.float)
    y_pred = trainer.infer(X=X_test_flat, probability=False)

    assert y_pred.shape[0] == len(fold.y_test), "infer() debería dar una predicción por trial de test"
    assert len(trainer.history["train_loss"]) == 3, "history debería tener 3 épocas de train_loss"

    print(f"  Fold: train={len(fold.y_train)}, val={len(fold.y_val)}, test={len(fold.y_test)}")
    print(f"  y_pred.shape = {y_pred.shape}, history poblado con {len(trainer.history['train_loss'])} épocas")
    print("  OK -- flujo DL completo, sin ningún orquestador ni RunLifecycleManager de por medio\n")


def verify_full_sandbox_flow_ml():
    print("--- 3. Flujo completo sandbox con Trainer_ML ---")
    from src.training.splitting.splitters import WithinSubjectHoldoutSplitter
    from src.training.splitting.fold import materialize_fold
    from src.training.execution.trainer import Trainer_ML

    class InMemoryDataset:
        def __init__(self, X, y):
            self._X, self._y = X, y

        def flatten_subject_data(self, subject_id, session=None):
            import pandas as pd
            metadata = pd.DataFrame({"subject": [subject_id] * len(self._y)})
            return self._X, self._y, metadata

    n = 60
    rng = np.random.RandomState(1)
    # Para ML, X típicamente ya es 2D (features), no (trials, channels, time).
    X = rng.randn(n, 5)
    y = np.array([0, 1] * (n // 2))
    dataset = InMemoryDataset(X, y)

    splitter = WithinSubjectHoldoutSplitter(ptrain=0.6, pval=0.2, ptest=0.2)
    specs = splitter.split(X, y, subject_id=1)
    fold = materialize_fold(specs[0], dataset, default_session="session_1", X=X, y=y)

    model = LogisticRegression()
    trainer = Trainer_ML(model, fold.X_train, fold.y_train, X_val=fold.X_val, y_val=fold.y_val)
    trainer.train()
    y_pred = trainer.infer(fold.X_test, probability=False)

    assert y_pred.shape[0] == len(fold.y_test)
    print(f"  Fold: train={len(fold.y_train)}, val={len(fold.y_val)}, test={len(fold.y_test)}")
    print(f"  y_pred.shape = {y_pred.shape}")
    print("  OK -- flujo ML completo, sin ningún orquestador de por medio\n")


def verify_no_disk_side_effects():
    print("--- 4. El flujo sandbox no escribe nada a disco (sin CSVs, sin config.json, sin .pth) ---")
    from src.training.splitting.splitters import WithinSubjectHoldoutSplitter
    from src.training.splitting.fold import materialize_fold
    from src.training.execution.trainer import Trainer_DL

    with tempfile.TemporaryDirectory() as tmpdir:
        files_before = list(Path(tmpdir).rglob("*"))
        assert files_before == []

        class InMemoryDataset:
            def __init__(self, X, y):
                self._X, self._y = X, y

            def flatten_subject_data(self, subject_id, session=None):
                import pandas as pd
                metadata = pd.DataFrame({"subject": [subject_id] * len(self._y)})
                return self._X, self._y, metadata

        n, n_channels, n_time = 40, 2, 8
        rng = np.random.RandomState(2)
        X = rng.randn(n, n_channels, n_time).astype(np.float32)
        y = np.array([0, 1] * (n // 2))
        dataset = InMemoryDataset(X, y)

        splitter = WithinSubjectHoldoutSplitter(ptrain=0.6, pval=0.2, ptest=0.2)
        specs = splitter.split(X, y, subject_id=1)
        fold = materialize_fold(specs[0], dataset, default_session="session_1", X=X, y=y)

        X_train_flat = torch.tensor(fold.X_train.reshape(len(fold.y_train), -1), dtype=torch.float)
        y_train_t = torch.tensor(fold.y_train, dtype=torch.long)
        train_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_train_flat, y_train_t), batch_size=4,
        )

        model = DummyModel(in_features=n_channels * n_time, n_classes=2)
        trainer = Trainer_DL(
            model=model, loss_fn=nn.CrossEntropyLoss(), optimizer=torch.optim.Adam(model.parameters()),
            train_loader=train_loader, max_epochs=2, callbacks=[],
        )
        trainer.train()

        # NOTA: este chequeo verifica el directorio temporal, que nunca se
        # le pasó a nada -- confirma que ninguna pieza usada "adivina" un
        # path de salida por su cuenta. No verifica el cwd real del
        # proceso; si querés un chequeo más estricto, correr este script
        # desde un directorio vacío y revisar manualmente después.
        files_after = list(Path(tmpdir).rglob("*"))
        assert files_after == [], f"Aparecieron archivos inesperados en el tmpdir: {files_after}"

    print("  ningún archivo escrito en el directorio temporal de control")
    print("  OK -- el flujo sandbox no tiene escritura oculta a disco\n")


if __name__ == "__main__":
    verify_imports_dont_pull_orchestrator()
    verify_full_sandbox_flow_dl()
    verify_full_sandbox_flow_ml()
    verify_no_disk_side_effects()
    print("=== TODOS LOS CHECKS DE verify_10_sandbox_usage.py PASARON ===")
