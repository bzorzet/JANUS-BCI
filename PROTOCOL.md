# Protocolo de trabajo — JANUS-BCI

Documento de referencia central. Si algo del código contradice
este documento, gana el documento — se actualiza el código o se
actualiza el documento explícitamente, nunca se ignora en silencio.

---

## 1. Las dos velocidades

| | Sandbox | Producción |
|---|---|---|
| Dónde vive | `sandbox/<project_name>/` | `src/` + `scripts/run_production.py` |
| ¿Trackeado por MLFlow/SQL? | No, nunca | Sí, siempre |
| ¿Puede leer la DB central? | Sí, solo lectura | Sí (es quien la escribe) |
| ¿Sigue el contrato de CSV? | No | Sí, obligatorio |
| Cómo se promueve | Reescritura manual como script de producción | — |

---

## 2. Taxonomía de nombres

Todo run se identifica con:

- **`project_name`** — ej. `2026-NID`
- **`strategy_name`** — semántica de validación: `WS-Standard`, `WS-LOSO`, `Across-Dataset`
- **`recipe_name`** — modelo o pipeline: `CTNet`, `FBCSP`, `CAR-preproc`
- **`script_type`** — `train_dl` | `train_ml` | `preprocessing` | `analysis` | `hypersearch` (futuro)
- **`context.source`** — dataset: `Cho2017`, `Lee2019`
- **`context.partition`** — unidad evaluada: `subject_08`
- **`context.replicate`** — corrida puntual: `split_8_seed_399`

**Regla de oro:** el path a disco *es* el esquema. Factores nuevos
(seed, fold, source_dataset...) son filas en `run_factors`, nunca
columnas nuevas en un CSV.

---

## 3. Estructura del repo

```
janus-bci/
├── .env                        # NO versionado — rutas de esta máquina
├── .env.example
├── .gitignore
├── PROTOCOL.md                 # este documento
├── SETUP.md                    # instalación desde cero
├── CLAUDE.md                   # instrucciones para Claude Code
├── README.md                   # punto de entrada general
│
├── Dockerfile                  # acepta ARG USE_GPU=true|false
├── docker-compose.yml          # profiles: gpu / cpu
├── environment.gpu.yml         # conda + CUDA 11.8
├── environment.cpu.yml         # conda + cpuonly
├── .devcontainer/
│   └── devcontainer.json       # VS Code Dev Containers
│
├── src/                        # código reutilizable entre proyectos
│   ├── utils/
│   │   └── paths.py            # objeto PATHS — único acceso a rutas
│   ├── models/                 # arquitecturas DL (CTNet, EEGNet...)
│   ├── eeg_datasets/           # clases de carga de datos
│   ├── torch_utils/            # DataLoader, callbacks, EarlyStopping
│   ├── training/               # loop de entrenamiento genérico
│   ├── preprocessing/          # pipelines de señal (CAR, ICA...)
│   └── analysis/               # métricas, estadísticas, espectral
│
├── projects/                   # un proyecto = un paper (o varios)
│   └── <project_name>/
│       ├── README.md           # descripción, datasets, modelos, comandos
│       ├── configs/
│       │   ├── training/
│       │   │   └── <strategy>/
│       │   │       └── <nombre>.json
│       │   └── preprocessing/
│       │       └── <nombre>.json
│       ├── generators/         # PRIVADO — no va al repo del paper
│       │   ├── config_generator.py
│       │   ├── models.py
│       │   ├── datasets.py
│       │   └── preprocessings.py
│       ├── analysis/           # scripts de análisis de resultados
│       └── commands/           # .sh para lanzar experimentos en batch
│
├── scripts/
│   ├── run_production.py       # entrypoint único para producción
│   ├── verify_env.py           # verifica el entorno Docker
│   └── sync/
│       ├── fetch_from_server.sh
│       ├── push_to_mlflow.py
│       └── build_analytics_db.py
│
├── db/
│   ├── schema.sql
│   ├── models.py
│   └── janus_analytics.db      # NO versionado, reconstruible
│
├── sandbox/
│   ├── README.md
│   ├── db_reader.py
│   └── <project_name>/         # estructura libre, descartable
│
├── mlflow/
│   └── mlflow.db               # NO versionado
│
└── tests/

```

---

## 4. Estructura de resultados (JANUS_RESULTS_ROOT — HDD)

```
RESULTS_ROOT/
└── <strategy_name>/
    └── <recipe_name>/
        └── <context.source>/
            ├── script_progress.csv        ← barrido completo
            └── <context.partition>/
                └── <context.replicate>/
                    ├── config.json        ← snapshot exacto del config usado
                    ├── .git_commit        ← hash del commit
                    ├── .docker_image      ← tag de la imagen Docker
                    ├── .mlflow_run_id     ← marcador de idempotencia
                    ├── metrics_results.csv
                    └── train_curve.csv    ← opcional, por época
```

---

## 5. Convención de nombrado de configs

### Training
```
{project}_{strategy}_{model}_{dataset}_{label}.json
```
Ejemplo:
```
2026-NID_WS-Standard_CTNet_Cho2017_CAR-Bilateral-Full.json
2026-NID_WS-LOSO_EEGNet_Lee2019_ICA10-Bilateral-Full.json
```

### Preprocessing
```
{dataset}_{session}_{preprocessing-name}.json
```
Ejemplo:
```
Cho2017_s1_CAR-preproc.json
Lee2019_s1_ICA10-preproc.json
```
Sin `project_name`: un mismo `{dataset}_{session}` puede tener varios
proyectos usando el mismo preprocesamiento, y el proyecto que lo generó
queda registrado en el `preprocessing_name` interno del JSON, no en el
nombre de archivo. Ver `preprocessing/DESIGN.md` sección 5.

**Reglas:**
- `_` separa campos, `-` separa palabras dentro de un campo
- El nombre del archivo es la identidad única del experimento —
  no pueden existir dos configs con el mismo nombre
- `script_type` siempre va explícito dentro del JSON
- El `label` es libre por proyecto — su semántica depende de qué
  se está ablacionando (canales, espectro, preprocesamiento, etc.)

---

## 6. Contratos de CSV

**`script_progress.csv`** — nivel barrido (`strategy/recipe/dataset/`):

| columna | descripción |
|---|---|
| `partition` | ej. `subject_08` |
| `status` | `pending` \| `running` \| `success` \| `failed` |
| `timestamp_start` | ISO 8601 |
| `timestamp_end` | ISO 8601 |

**`metrics_results.csv`** — nivel hoja (`context.replicate/`):

| columna | descripción |
|---|---|
| `split` | `train` \| `val` \| `test` |
| `metric_name` | ej. `accuracy`, `f1` |
| `value` | numérico |

---

## 7. Base de datos analítica

- 100% reconstruible: `build_analytics_db.py` borra y repuebla
  desde los CSVs en disco. Nunca se edita a mano.
- Tablas: `runs`, `metrics`, `run_factors` (todas en formato largo)
- Sandbox: solo lectura vía `sandbox/db_reader.py` (SQLite `mode=ro`)

---

## 8. Reproducibilidad radical

Cada corrida deja en su carpeta hoja el trío:
1. `config.json` — snapshot exacto del JSON usado
2. `.git_commit` — hash del commit
3. `.docker_image` — tag de la imagen Docker

Sin estos tres datos, un resultado no se considera trazable.

---

## 9. Flujo de sincronización con servidores

```
fetch_from_server.sh  →  push_to_mlflow.py  →  build_analytics_db.py
(rsync, trae archivos)   (idempotente vía        (reconstruye SQL
                          .mlflow_run_id)          desde RESULTS_ROOT)
```

Siempre en ese orden, nunca salteado.

---

## 10. Generadores de configs (privados)

Cada proyecto tiene su `generators/` con el script que produce
los JSONs automáticamente combinando datasets × modelos ×
preprocesamiento. Esta carpeta:
- **No se versiona en el repo público del paper**
- Se agrega a `.gitignore` antes de publicar: `projects/*/generators/`
- Los JSONs generados sí se versionan y publican

---

## 11. Checklist antes de arrancar un proyecto nuevo

1. ¿Definiste `project_name` y las `strategy_name` que vas a usar?
2. ¿El preprocesamiento tiene nombre versionado
   (`{project}_{dataset}_{session}_{prep-name}.json`)?
3. ¿Hay factores nuevos para `run_factors`? Si es así, agregá el
   parser en `build_analytics_db.py`.
4. ¿Creaste `projects/<nombre>/README.md` con datasets, modelos
   y convención de label?
5. ¿Vas a probar algo primero? Andá a `sandbox/<project>/`.

---


## 12. Ítems abiertos

- **`hypersearch`** como `script_type` futuro — se define cuando
  haya un caso real de búsqueda de hiperparámetros.
- **Bootstrap automático de dominio nuevo** (imágenes, PINNs) —
  extraer `janus-core-base` cuando exista un segundo dominio real.
- **Windows / Mac** en `SETUP.md` — pendiente hasta poder probarlo.
- **Reglas de git** (branches, qué se commitea) — no definidas aún.
- **Dev Containers** — descartado para desarrollo diario por problemas
  de compatibilidad de versiones entre sesiones SSH y físicas.
  Flujo adoptado: VS Code Remote SSH + `BCI_decoding_env` para
  desarrollo, Docker solo para producción y servidores. Ver `BUGS.md`.