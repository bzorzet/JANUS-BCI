"""
Resolución dinámica de clases/funciones por nombre + módulo.

Usado por el motor de stages (`src/preprocessing/preprocess_eeg.py`) para
instanciar objetos declarados en JSON (ej. `mne.preprocessing.ICA`,
`src.preprocessing.normalize_by_mean_std`) sin tener que hardcodear un
import por cada posible step.
"""
import importlib


def import_class(name: str, module_name: str):
    module = importlib.import_module(module_name)
    return getattr(module, name)
