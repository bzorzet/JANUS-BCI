# Rediseño de carga y preprocesamiento — JANUS-BCI

Documento de diseño. Complementa a `PROTOCOL.md` y sigue sus mismas reglas:
si algo de acá contradice al código o a `PROTOCOL.md`, se actualiza
explícitamente, nunca se ignora en silencio.

Este documento cubre las Capas 2 a 5 del preprocesamiento (Subject, motor de
pipeline, orquestador, lectura). La Capa 1 (loaders de dataset: `Cho2017`,
`Dreyer2023A/B/C`) queda fuera de esta fase — ver sección 8.

---

## 0. Alcance

Reemplaza:
- `src/preprocessing/preprocess_eeg.py` (`EEGPreprocessor`)
- `src/preprocessing/database_preprocessor.py` (`EEGDatabasePreprocessor`)
- `src/ica/ica_processor.py` (`ICAProcessor`, repo viejo) — se absorbe como stage handler
- `src/ica/ica_database_preprocessor.py` (`EEGDatabaseICAPreprocessor`, repo viejo) — se elimina como clase separada

Corrige (no reescribe de cero):
- `src/eeg_datasets/preprocessed_dataset.py` (`PreprocessedDataset`)

**Actualización:** la Capa 1 (`base.py`, `subject.py`, `cho2017.py`,
`dreyer2023.py`, más `lee2019.py`) dejó de estar fuera de alcance — se
migró junto con las Fases 2 y 3 a pedido explícito del usuario, antes de
lo previsto en la sección 9. Ver sección 8 por las decisiones tomadas.

## 1. Motivación

`EEGDatabaseICAPreprocessor` es hoy una copia casi completa de
`EEGDatabasePreprocessor` con ICA insertada en el medio. Esto ya generó
deriva real: `data_availables` vs `data_to_load` en `_save_dataset_metadata`,
el `try/except` de `get_subject` solo en la versión ICA, y un tipo de
retorno inconsistente en `ICAProcessor.process()` (a veces `Raw`, a veces
`dict`). El objetivo es que una forma nueva de preprocesar (ASR, ventaneo,
lo que sea) se agregue registrando una función, no escribiendo otra clase
que reimplementa la orquestación completa.

## 2. Motor de pipeline — `EEGPreprocessor`

Config-driven. Recibe una lista ordenada de *stages*. Cada stage se resuelve
contra un registro de handlers por `stage_type`.

```python
@dataclass
class StageResult:
    data: Any                                    # pasa al siguiente stage
    metrics: dict[str, float | int | str] = {}   # escalares -> stage_metrics.csv
    detail_tables: dict[str, list[dict]] = {}    # registros -> <stage>_<key>.csv
    artifacts: dict[str, Any] = {}                # objetos pesados -> artifact saver
```

```python
STAGE_HANDLERS = {
    "steps": handle_steps,        # envuelve la _apply_steps ya existente
    "ica": handle_ica,            # migrado de ica_processor.py
    "epoching": handle_epoching,  # envuelve la lógica de Epochs ya existente
}
```

`EEGPreprocessor.register_stage_handler(stage_type, fn)` permite extender
sin tocar la clase. `process(raw)` ejecuta los stages en orden, encadena
`data`, y acumula `metrics`/`detail_tables`/`artifacts` de todos los stages
en un único `StageResult` final. Un stage `"steps"` que no produce nada
especial simplemente devuelve `StageResult(data=nuevo_dato)`.

## 3. Esquema de config JSON

```json
{
  "preprocessing_name": "2026-NID-10ICA-cleaned-preprocessing",
  "database": { "class_name": "Cho2017", "module_name": "src.eeg_datasets",
                "kwargs": { "sessions": ["session_1"], "subjects": null,
                            "data_to_load": ["motor_imagery"] } },
  "preprocessing_pipeline": {
    "stages": [
      {"stage_name": "raw_filter_car", "stage_type": "steps", "target": "raw",
       "steps": [ {"type": "method", "name": "filter", "kwargs": {"l_freq": 1.0, "h_freq": 100.0}},
                  {"type": "method", "name": "notch_filter", "kwargs": {"freqs": [60.0]}},
                  {"type": "method", "name": "set_eeg_reference", "kwargs": {"ref_channels": "average"}} ]},
      {"stage_name": "ica_cleaning", "stage_type": "ica", "target": "raw",
       "config": { "ica_instance": {"class_name": "ICA", "module_name": "mne.preprocessing",
                                     "kwargs": {"n_components": 10, "method": "infomax", "random_state": 97}},
                    "ica_fit": {"picks": "eeg", "reject_by_annotation": true},
                    "ica_label": {"method": "iclabel", "labels_to_keep": ["brain", "other"]} }},
      {"stage_name": "raw_filter_post_ica", "stage_type": "steps", "target": "raw",
       "steps": [ {"type": "method", "name": "filter", "kwargs": {"l_freq": 1.0, "h_freq": 40.0}} ]},
      {"stage_name": "epoching", "stage_type": "epoching",
       "config": {"event_id": {"left_hand": 1, "right_hand": 2}, "tmin": -1.5, "tmax": 3.5, "decim": 4}}
    ]
  },
  "description": "ICA preprocessing pipeline for MI-EEG data"
}
```

Un preprocesamiento simple (sin ICA) es la misma estructura con menos
stages — no hay un formato "simple" y otro "ICA" distintos.

## 4. Orquestador — `EEGDatabasePreprocessor` (uno solo)

Itera `dataset → subjects → sessions → runs`. Por cada run:

1. `worker.process(raw)` → `StageResult`.
2. Guarda `data` con la misma lógica de siempre (`Raw`→`.fif`, `Epochs`→`.epo.fif`, `dict`→`.npy`).
3. Por cada key en `artifacts`, busca un saver en `ARTIFACT_SAVERS[key]`; si no hay
   uno registrado, loguea un aviso y sigue — nunca rompe el barrido.
4. Agrega una fila a `run_registry.csv`, filas a `stage_metrics.csv`, filas a
   cada `<stage>_<detail_key>.csv` que corresponda — **en modo append, con
   flush por corrida** (no acumular en memoria hasta el final: si el barrido
   se cae en el sujeto 40, los primeros 39 deben quedar consultables).
5. Escribe el trío de reproducibilidad (`config.json`, `.git_commit`,
   `.docker_image`) **una sola vez, al inicio del barrido** (antes del loop),
   junto con el `dataset_description.json` inicial.

## 5. Ubicación de archivos

**Config (repo, no versiona datos):**
```
preprocessing/configs/<dataset>/<dataset>_<session>_<recipe-name>.json
```
Ejemplo: `preprocessing/configs/Cho2017/Cho2017_s1_10ICA-cleaned-preproc.json`

**Salida (JANUS_DATA_ROOT=/data/EEG_DATABASES):**
```
/data/EEG_DATABASES/preprocessed/<preprocessing_name>/<dataset_code>/
├── config.json                             # trío 1/3
├── .git_commit                             # trío 2/3
├── .docker_image                           # trío 3/3
├── dataset_description.json                # manifest, lo lee PreprocessedDataset
├── run_registry.csv
├── stage_metrics.csv
├── <stage_name>_<detail_key>.csv           # uno por cada detail_table que aporte algún stage
├── artifacts/
│   └── <stage_name>/
│       └── subject_XX/
│           └── ...                         # ej. ica: .fif del ICA + pngs de auditoría
└── <session>/
    └── subject_XX/
        └── <prefix>_{raw|epo}.fif  o  <prefix>_{data|labels}.npy
```

Cambio necesario en `src/utils/paths.py`: agregar `PATHS.preprocessed_root`
apuntando a `<JANUS_DATA_ROOT>/preprocessed`.

## 6. Esquema de los 3 CSV (formato largo, ver PROTOCOL.md sección 6)

**`run_registry.csv`**: `partition, subject_id, session, context, run_id, status, timestamp_start, timestamp_end, output_path`

**`stage_metrics.csv`**: `partition, stage_name, metric_name, value`

**`<stage_name>_<detail_key>.csv`** (ej. `ica_cleaning_component_labels.csv`): `partition, component, label, probability, excluded`
(las columnas después de `partition` son propias de cada `detail_table` — no fijas)

`partition` es un string tipo `subject_08/motor_imagery/run_1`, consistente
entre los tres archivos para poder cruzarlos por join.

## 7. Lectura — `PreprocessedDataset` y `PreprocessingReport`

**Fix necesario en `PreprocessedDataset`** (bug preexistente, no depende del
resto del rediseño): hoy `PreprocessedDataset(db_name, ...)` no recibe
`preprocessing_name`, así que no puede distinguir dos recetas distintas para
el mismo dataset. Nueva firma:

```python
PreprocessedDataset(preprocessing_name: str, db_name: str,
                     sessions=None, subjects=None, data_to_load=None,
                     channels=None, classes_to_return=None)
```
con `self.base_path = os.path.join(PATHS.preprocessed_root, preprocessing_name, db_name)`.
El resto de la clase (`_load_metadata`, `get_subject`, `flatten_subject_data`,
`flatten_pool_data`) no cambia de lógica.

**Nueva clase, mismo módulo**, de solo lectura (mismo espíritu que
`sandbox/db_reader.py`):

```python
class PreprocessingReport:
    def __init__(self, preprocessing_name: str, db_name: str): ...
    def run_registry(self) -> pd.DataFrame: ...
    def stage_metrics(self, stage_name=None, metric_name=None) -> pd.DataFrame: ...
    def detail_table(self, stage_name: str, detail_key: str) -> pd.DataFrame: ...
```

No toca la carga de señal — solo lee los CSV de la sección 6. Es la vía para
responder preguntas de análisis futuro (ej. cantidad de fuentes etiquetadas
como ruido y de qué tipo, vía `detail_table("ica_cleaning","component_labels")`).

## 8. Ítems resueltos / abiertos

- **Capa 1 — unificación de `Dreyer2023A/B/C`: RESUELTO.** Se optó por
  clase base + subclases finitas: `Dreyer2023Base` concentra
  `_create_raw_simple`, `_obtain_extra_info`, `get_subject` y `download`;
  `Dreyer2023A/B/C` solo declaran `FOLDER_PREFIX`, `DATA_SUBDIR`,
  `METADATA_SUFFIX`, `SUBJECT_RANGE` y `DATA_AVAILABLES`. Al unificar bajo
  `FOLDER_PREFIX`, el bug de `Dreyer2023B.get_subject` (buscaba `.mat` en
  vez de `.gdf`/carpeta `B{id}`) desaparece solo — ver
  `src/eeg_datasets/dreyer2023.py`.
- **`download()` de datasets: RESUELTO (parcialmente).** Interfaz
  definida: `BaseEEGDataset.download()` es abstracto. Implementación real
  todavía no — cada dataset (`Cho2017`, `Dreyer2023*`, `Lee2019`) tiene un
  stub que levanta `NotImplementedError` con instrucciones, a la espera de
  que se confirme la fuente exacta de descarga (GigaDB para Cho2017/
  Lee2019, Sci Data/Zenodo para Dreyer2023) en una iteración futura — no se
  inventó ninguna URL de mirror sin verificar.
- `SessionState`, mencionado en el docstring de `preprocess_eeg.py` actual —
  confirmar si `StageResult` ya cubre ese rol o si era una idea distinta.

## 9. Plan de migración (secuencial)

| Fase | Contenido | Depende de | Estado |
|---|---|---|---|
| 1 | Motor: `StageResult`, registro de stage handlers, migración de ICA a handler, migración de los 4 JSON de ejemplo al esquema de `stages` | — | Hecho |
| 2 | Orquestador único, registro de artifact savers, CSVs incrementales, trío | Fase 1 | Hecho |
| 3 | `PreprocessedDataset` + `PreprocessingReport` | Fase 2 (para tener datos reales que leer) | Hecho |
| 4 | Capa 1 (loaders) — decisión de la sección 8 ya tomada | independiente | Hecho (`download()` real pendiente, ver sección 8) |

Fases 2-4 se hicieron juntas, adelantadas respecto al orden original, a
pedido explícito del usuario.