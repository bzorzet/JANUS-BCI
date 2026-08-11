"""
Motor de pipeline de preprocesamiento — config-driven, basado en stages.

Ver `preprocessing/DESIGN.md` sección 2. `EEGPreprocessor` recibe una lista
ordenada de stages (`preprocessing_pipeline["stages"]` del config JSON) y
ejecuta cada uno contra un handler resuelto por `stage["stage_type"]` en
`STAGE_HANDLERS`. Cada handler devuelve un `StageResult`; `process()`
encadena el `data` de un stage al siguiente y acumula `metrics`,
`detail_tables` y `artifacts` de todos los stages, anidados por
`stage_name`, en un único `StageResult` final.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import mne

from src.utils.imports import import_class


@dataclass
class StageResult:
    data: Any
    metrics: Dict[str, Any] = field(default_factory=dict)
    detail_tables: Dict[str, List[dict]] = field(default_factory=dict)
    artifacts: Dict[str, Any] = field(default_factory=dict)


def _apply_single_step(data, step: dict):
    step_type = step.get("type")
    name = step["name"]
    module_name = step.get("module_name")
    kwargs = step.get("kwargs", {}).copy()

    if step_type == "method":
        if not hasattr(data, name):
            raise AttributeError(f"Object {type(data)} has no method '{name}'")
        result = getattr(data, name)(**kwargs)
        # MNE methods often modify in-place and return self, but some
        # (e.g. constructors reached via 'method') return a new object.
        if result is not None:
            data = result
    elif step_type in ("function", "class", "class_init"):
        fn = import_class(name, module_name)
        data = fn(data, **kwargs)
    else:
        raise ValueError(f"Unknown step type '{step_type}'")
    return data


def handle_steps(data, stage: dict) -> StageResult:
    for step in stage.get("steps", []):
        data = _apply_single_step(data, step)
    return StageResult(data=data)


def handle_epoching(data, stage: dict) -> StageResult:
    config = dict(stage.get("config", {}))
    events = mne.find_events(data, stim_channel=None, verbose=False)
    if len(events) == 0:
        raise ValueError("No events found in data. Cannot epoch.")
    config["events"] = events
    epochs = mne.Epochs(data, **config)
    return StageResult(data=epochs)


def handle_ica(data, stage: dict) -> StageResult:
    from mne_icalabel import label_components  # deferred: optional heavy dep

    config = stage["config"]

    ica_cls = import_class(
        config["ica_instance"]["class_name"], config["ica_instance"]["module_name"]
    )
    ica = ica_cls(**config["ica_instance"]["kwargs"])

    raw_before_ica = data.copy()
    ica.fit(data, **config.get("ica_fit", {}))

    ica_labels = label_components(inst=data, ica=ica, method=config["ica_label"]["method"])
    labels = ica_labels["labels"]
    probabilities = ica_labels["y_pred_proba"]
    labels_to_keep = set(config["ica_label"]["labels_to_keep"])
    exclude_idx = [i for i, label in enumerate(labels) if label not in labels_to_keep]

    cleaned_raw = data.copy()
    ica.apply(cleaned_raw, exclude=exclude_idx)

    component_labels = [
        {
            "component": i,
            "label": label,
            "probability": float(probabilities[i]),
            "excluded": i in exclude_idx,
        }
        for i, label in enumerate(labels)
    ]

    return StageResult(
        data=cleaned_raw,
        metrics={
            "n_components_total": len(labels),
            "n_components_excluded": len(exclude_idx),
        },
        detail_tables={"component_labels": component_labels},
        artifacts={"ica_obj": ica, "raw_before_ica": raw_before_ica},
    )


STAGE_HANDLERS: Dict[str, Callable[[Any, dict], StageResult]] = {
    "steps": handle_steps,
    "ica": handle_ica,
    "epoching": handle_epoching,
}


class EEGPreprocessor:
    """
    Ejecuta `stages` (lista ordenada de dicts, ver DESIGN.md sección 3) en
    orden contra un registro de handlers por `stage_type`.
    """

    def __init__(self, stages: Optional[List[dict]] = None):
        self.stages = stages if stages is not None else []
        # Copia de instancia: registrar un handler acá no pisa el registro
        # global ni afecta a otras instancias.
        self._stage_handlers: Dict[str, Callable[[Any, dict], StageResult]] = dict(
            STAGE_HANDLERS
        )

    def register_stage_handler(self, stage_type: str, fn: Callable[[Any, dict], StageResult]):
        self._stage_handlers[stage_type] = fn

    def process(self, raw) -> StageResult:
        data = raw.copy()
        metrics: Dict[str, dict] = {}
        detail_tables: Dict[str, dict] = {}
        artifacts: Dict[str, dict] = {}

        for stage in self.stages:
            stage_type = stage["stage_type"]
            handler = self._stage_handlers.get(stage_type)
            if handler is None:
                raise ValueError(f"No handler registered for stage_type '{stage_type}'")

            result = handler(data, stage)
            data = result.data

            stage_name = stage.get("stage_name", stage_type)
            if result.metrics:
                metrics[stage_name] = result.metrics
            if result.detail_tables:
                detail_tables[stage_name] = result.detail_tables
            if result.artifacts:
                artifacts[stage_name] = result.artifacts

        return StageResult(
            data=data, metrics=metrics, detail_tables=detail_tables, artifacts=artifacts
        )
