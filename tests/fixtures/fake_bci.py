"""
Dataset/modelo sintéticos usados por tests/test_training_orchestrator.py.
Stand-in por PreprocessedDataset/arquitecturas reales de src/networks/
(migración de arquitecturas fuera de alcance de este refactor -- ver
decisión 6 del plan) para poder ejercitar el orquestador de training de
punta a punta sin depender de datos EEG reales.
"""
import numpy as np
import pandas as pd
import torch


class FakeDataset:
    def __init__(self, n_subjects=2, n_trials=40, n_channels=4, n_times=16):
        self.sessions = ["session_1"]
        self.subject_list = list(range(1, n_subjects + 1))
        self.n_trials = n_trials
        self.n_channels = n_channels
        self.n_times = n_times

    def flatten_subject_data(self, subject_id, session=None):
        rng = np.random.RandomState(subject_id)
        X = rng.randn(self.n_trials, self.n_channels, self.n_times).astype(np.float32)
        y = np.array(([0, 1] * (self.n_trials // 2))[:self.n_trials])
        metadata = pd.DataFrame({"trial": range(self.n_trials)})
        return X, y, metadata


class TinyClassifier(torch.nn.Module):
    def __init__(self, n_channels, n_times, n_classes=2):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(n_channels * n_times, n_classes)

    def forward(self, x):
        return self.linear(self.flatten(x))
