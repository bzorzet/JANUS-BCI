"""
Funciones puras migradas verbatim del script viejo
(repo_viejo/train_model_within_subject_online_simulated_for_MI-BCI_classification.py
y repo_viejo/src/utils/auxiliary_functions.py). Usadas por el orquestador de
training (src/training/orchestrator.py) para instanciar datasets/dataloaders/
callbacks desde config ya resuelto -- ninguna de estas funciones lee un JSON
de config por su cuenta, reciben todo inyectado (principio 2 del refactor).
"""
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.metrics import accuracy_score

from src.utils.imports import import_class


def accuracy(y_pred, y_true) -> float:
    """Score function esperado por EpochScoring: score_fn(y_pred, y_true).
    Si y_pred viene en 2D (logits/probabilidades), argmax primero."""
    if len(y_pred.shape) == 2:
        y_pred = np.argmax(y_pred, axis=1)
    return accuracy_score(y_true, y_pred)


def create_dataset(split_name: str, X, y, ds_config: dict):
    """Instancia un torch Dataset vía import_class. `ds_config` puede tener
    una entrada por split (train/val/test) o ser compartido -- si
    `split_name` es una key de `ds_config`, se usa esa sub-config, si no se
    usa `ds_config` directamente."""
    if split_name in ds_config:
        config = ds_config[split_name]
    else:
        config = ds_config
    config['params']['X'] = X
    config['params']['y'] = y
    return import_class(config['class_name'], config['module_name'])(**config['params'])


def create_dataloader(split_name: str, dataset, dl_config: dict, g):
    """Instancia un DataLoader vía import_class. Mismo patrón per-split-o-
    compartido que create_dataset. `g` es el torch.Generator compartido de
    todo el barrido (seedeado una vez con general_script_config.seed)."""
    if split_name in dl_config:
        config = dl_config[split_name]
    else:
        config = dl_config
    config['params']['dataset'] = dataset
    config['params']['generator'] = g
    return import_class(config['class_name'], config['module_name'])(**config['params'])


def build_callbacks(
    base_callbacks: Optional[List[Any]] = None,
    config_callbacks: Optional[List[dict]] = None,
    aditional_params: Optional[Dict[str, Any]] = None,
) -> List[Any]:
    """Combina callbacks Python ya instanciados (`base_callbacks`) con
    callbacks armados desde config (`config_callbacks`, lista de dicts
    class_name/module_name/params). A cualquier callback de config cuyos
    params tengan una key `dirname`, se le inyecta
    `aditional_params["path_to_save"]`.

    Nota de migración: el original leía una variable global `path_to_save`
    (en vez de `aditional_params["path_to_save"]`) para decidir SI inyectar
    -- funcionaba solo porque esa global coincidía con
    `aditional_params["path_to_save"]` en el único caller real. Acá, sin
    ese acoplamiento al scope del script, la condición usa directamente
    `aditional_params` -- mismo comportamiento observable, sin la
    dependencia de una global inexistente en este módulo."""
    base_callbacks = base_callbacks or []
    final_callbacks = list(base_callbacks)
    if config_callbacks:
        for cb in config_callbacks:
            cb_params = cb.get('params', {}).copy()
            if 'dirname' in cb_params and aditional_params is not None and aditional_params.get("path_to_save") is not None:
                cb_params['dirname'] = aditional_params["path_to_save"]
            cb_instance = import_class(cb['class_name'], cb['module_name'])(**cb_params)
            final_callbacks.append(cb_instance)
    return final_callbacks


def convert_labels_to_int(labels, dict_labels: Optional[dict] = None):
    """Puerto verbatim de repo_viejo/src/utils/auxiliary_functions.py."""
    dict_labels = dict_labels or {}
    return np.vectorize(dict_labels.get)(labels)


def encode_labels(y, label_map: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Mapea las clases de y a enteros consecutivos 0..N-1, preservando
    el orden de aparición de np.unique(y) si no se pasa label_map. Sirve
    para cualquier N >= 2 -- el caso binario (N=2) es un caso particular,
    no una restricción del contrato.

    label_map, si se pasa, fuerza el mapeo (útil para que train/val/test
    usen EXACTAMENTE el mismo mapeo, en vez de que cada split calcule el
    suyo con np.unique -- ver la nota de bin_to_class consistente ya
    marcada como pendiente en el prompt de splitters)."""
    if label_map is None:
        unique_classes = np.unique(y)
        label_map = {cls: i for i, cls in enumerate(unique_classes)}
    y_encoded = np.vectorize(label_map.get)(y)
    return y_encoded, label_map
