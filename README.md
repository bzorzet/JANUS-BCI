# JANUS-BCI

Framework de experimentación para Deep Learning aplicado a BCI
(Brain-Computer Interfaces). Diseñado para resolver la tensión entre
**prototipado rápido** y **producción reproducible y trackeable**.

Pensado para un flujo de investigación de PhD: múltiples proyectos,
múltiples datasets, múltiples modelos, corridas en laptop y en
servidores del instituto — todo gestionado desde un único repo.

---

## Documentación

| Documento | Contenido |
|---|---|
| [`SETUP.md`](SETUP.md) | Instalación desde cero: Docker, GPU, primer build |
| [`PROTOCOL.md`](PROTOCOL.md) | Reglas de trabajo, taxonomía, estructura de archivos y convención de nombrado |
| [`CLAUDE.md`](CLAUDE.md) | Referencia rápida para Claude Code: comandos y reglas |
| [`BUGS.md`](BUGS.md) | Bitácora cronológica de problemas reales y sus soluciones |

---

## Estructura general

```
janus-bci/
│
├── src/                    # Código reutilizable entre proyectos
│   ├── networks/           # Arquitecturas DL: CTNet, EEGNet, ATCNet...
│   ├── eeg_datasets/       # Carga de datasets EEG
│   ├── torch_utils/        # DataLoaders, callbacks, EarlyStopping
│   ├── training/           # Loop de entrenamiento genérico
│   ├── preprocessing/      # Pipelines de señal: CAR, ICA, filtros
│   ├── analysis/           # Métricas, estadísticas, análisis espectral
│   └── utils/paths.py      # Único punto de acceso a rutas (objeto PATHS)
│
├── projects/               # Un proyecto = un paper (o varios papers)
│   └── NID/                # → ver projects/NID/README.md
│
├── scripts/
│   ├── run_production.py   # Entrypoint único para cualquier corrida
│   └── sync/               # Sincronización con servidores del instituto
│
├── sandbox/                # Prototipado rápido — aislado, descartable
│
├── db/                     # DB analítica reconstruible (SQLite)
├── mlflow/                 # Backend de MLFlow
│
├── Dockerfile              # Imagen Docker (GPU o CPU según ARG)
├── docker-compose.yml      # Profiles: gpu / cpu
├── environment.gpu.yml     # Dependencias con CUDA 11.8
└── environment.cpu.yml     # Dependencias sin GPU
```

---

## Las dos velocidades

JANUS-BCI está diseñado para dos modos que **no se mezclan**:

### Sandbox — para probar ideas rápido
```
sandbox/<project_name>/
```
- Estructura libre, sin contrato de CSV, sin tracking
- Puede leer la DB central (solo lectura)
- Nunca escribe en MLFlow ni en la DB analítica
- Cuando la idea funciona → se reescribe como script de producción

### Producción — para resultados trackeables
```
python scripts/run_production.py --config projects/<project>/configs/...json
```
- Todo queda trackeado en MLFlow y en la DB analítica
- Sigue el contrato de CSV (`script_progress.csv`, `metrics_results.csv`)
- Deja el trío de reproducibilidad en cada carpeta de resultados
  (`config.json`, `.git_commit`, `.docker_image`)

---

## Convención de nombrado de configs

El nombre del archivo JSON es la identidad única del experimento.

**Training:**
```
{project}_{strategy}_{model}_{dataset}_{label}.json

2026-NID_WS-Standard_CTNet_Cho2017_CAR-Bilateral-Full.json
```

**Preprocessing:**
```
{project}_{dataset}_{session}_{preprocessing-name}.json

2026-NID_Cho2017_s1_CAR-preproc.json
```

`_` separa campos, `-` separa palabras dentro de un campo.
Ver [`PROTOCOL.md`](PROTOCOL.md) sección 5 para la convención completa.

---

## Estructura de resultados

Los resultados **no viven en el repo** — viven en `JANUS_RESULTS_ROOT`
(HDD, definido en `.env`):

```
RESULTS_ROOT/
└── <strategy>/
    └── <model>/
        └── <dataset>/
            ├── script_progress.csv     ← tracking del barrido completo
            └── <subject>/
                └── <replicate>/
                    ├── config.json     ← reproducibilidad
                    ├── .git_commit
                    ├── .docker_image
                    ├── metrics_results.csv
                    └── train_curve.csv
```

---

## Proyectos

| Proyecto | Descripción | Estado |
|---|---|---|
| [NID](projects/NID/README.md) | Clasificación MI-BCI, comparación DL vs ML, ablación espacial y espectral | En desarrollo |

Para crear un proyecto nuevo ver el checklist en
[`PROTOCOL.md`](PROTOCOL.md) sección 11.

---

## Flujo de trabajo diario

```bash
# Abrir el entorno en VS Code Dev Containers
# Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# Preprocesar un dataset
python scripts/run_production.py \
  --config projects/NID/configs/preprocessing/2026-NID_Cho2017_s1_CAR-preproc.json

# Entrenar un modelo
python scripts/run_production.py \
  --config projects/NID/configs/training/WS-Standard/2026-NID_WS-Standard_CTNet_Cho2017_CAR-Bilateral-Full.json

# Ver resultados en MLFlow
docker compose up mlflow   # → http://localhost:5000

# Traer resultados del servidor del instituto
bash scripts/sync/fetch_from_server.sh
python scripts/sync/push_to_mlflow.py
python scripts/sync/build_analytics_db.py
```

---

## Setup

Ver [`SETUP.md`](SETUP.md) para la guía completa de instalación.
Resumen:

```bash
git clone <repo> janus-bci && cd janus-bci
cp .env.example .env          # completar rutas locales
docker compose --profile gpu build bci-gpu
docker compose --profile gpu run --rm bci-gpu python scripts/verify_env.py
```
