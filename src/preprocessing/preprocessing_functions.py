"""
Funciones invocables desde un step `"type": "function", "module_name":
"src.preprocessing"` en un config de preprocessing. Se portan una por una
desde `repo_viejo/src/preprocessing/preprocessing_functions.py` a medida
que algún config las use — no se copia el archivo entero de una.
"""
import mne
import numpy as np


def extract_data_from_epochs(epochs, **kwargs):
    """
    Extrae data, labels y metadata de un objeto Epochs.

    Parameters
    ----------
    epochs : mne.Epochs
    **kwargs :
        Argumentos pasados a epochs.get_data() (ej. units, tmin, tmax).

    Returns
    -------
    dict
        {'data': np.array, 'labels': np.array, 'ch_names': list, 'sfreq': float}
    """
    result = {
        "sfreq": epochs.info["sfreq"],
        "ch_names": epochs.ch_names,
        "data": epochs.get_data(**kwargs),
        "event_id": epochs.event_id,
    }
    if epochs.events is not None:
        result["labels"] = epochs.events[:, -1]
    else:
        result["labels"] = np.array([])
    return result


def normalize_by_mean_std(data):
    eps = 1e-10
    if isinstance(data, dict) and "data" in data:
        data["data"] = (data["data"] - data["data"].mean(axis=-1, keepdims=True)) / (
            data["data"].std(axis=-1, keepdims=True) + eps
        )
        return data
    elif isinstance(data, mne.Epochs):
        data_array = data.get_data(copy=True)
        norm_data = (data_array - data_array.mean(axis=-1, keepdims=True)) / (
            data_array.std(axis=-1, keepdims=True) + eps
        )
        return mne.EpochsArray(
            norm_data, data.info, events=data.events, event_id=data.event_id, tmin=data.tmin
        )
    elif isinstance(data, mne.io.BaseRaw):
        # Puerto fiel del original: no retorna nada para Raw (no lo usa
        # ningún config migrado, que solo llama a esto post-epoching).
        data_array = data.get_data()
        norm_data = (data_array - data_array.mean(axis=-1, keepdims=True)) / (
            data_array.std(axis=-1, keepdims=True) + eps
        )
    else:
        raise TypeError("Data type not supported for normalization.")
