# CLAUDE.md

Instrucciones de proyecto para JANUS-BCI. Se cargan automáticamente al
empezar cada sesión de Claude Code.

@PROTOCOL.md

## Setup rápido

- Copiar `.env.example` a `.env` y completar las rutas de esta máquina
  (nunca commitear `.env`).
- `docker compose build`.
- `docker compose up mlflow` levanta el dashboard en `localhost:5000`.

## Comandos

### Docker

| Comando | Qué hace |
|---|---|
| `docker compose --profile gpu build bci-gpu` | construye la imagen con CUDA (primera vez o tras cambiar `environment.gpu.yml`) |
| `docker compose --profile cpu build bci-cpu` | construye la imagen sin GPU (colaboradores, CI) |
| `docker compose --profile gpu run --rm bci-gpu bash` | shell interactivo con GPU |
| `docker compose --profile cpu run --rm bci-cpu bash` | shell interactivo sin GPU |
| `docker compose up mlflow` | dashboard de MLFlow en `localhost:5000` |

### Dentro del contenedor (o con `-v` en desarrollo)

| Comando | Qué hace |
|---|---|
| `python scripts/run_production.py --config <path>` | corre un barrido de producción completo |
| `python scripts/sync/build_analytics_db.py` | reconstruye `db/janus_analytics.db` desde `RESULTS_ROOT` |
| `bash scripts/sync/fetch_from_server.sh` | trae resultados nuevos de un server del instituto |
| `python scripts/sync/push_to_mlflow.py` | sube a MLFlow las corridas nuevas (idempotente) |
| `python scripts/verify_env.py` | verifica que el entorno tiene todas las librerías |

## Reglas que nunca hay que romper

- No escribir a `db/janus_analytics.db` desde ningún lado excepto
  `scripts/sync/build_analytics_db.py`. Cualquier lectura — incluido
  todo lo que vive bajo `sandbox/` — pasa por `get_readonly_session()`
  de `db/models.py` o `sandbox/db_reader.py`.
- No hardcodear rutas. Todo path sale de `PATHS`
  (`src/utils/paths.py`), nunca de `os.environ` directo ni de un
  string literal.
- Todo script de producción nuevo escribe `script_progress.csv` y
  `metrics_results.csv` con las columnas exactas de `PROTOCOL.md`
  sección 5 — no inventar nombres de columna nuevos.
- Un factor de contexto nuevo (seed, fold, source_dataset...) es una
  fila en `run_factors`, nunca una columna nueva en un CSV o en
  `schema.sql`.
- Código de prueba rápida va en `sandbox/<project_name>/` y no toca
  MLFlow ni escribe en `db/janus_analytics.db`, nunca.
- Antes de escribir un script de producción nuevo, confirmar
  `project_name` / `strategy_name` / `recipe_name` / `script_type` /
  `context.*` (PROTOCOL.md sección 2) — son los tags que después
  arman el path en `RESULTS_ROOT`.

## Dónde está cada cosa

- `src/utils/paths.py` — único punto de acceso a rutas (objeto `PATHS`).
- `db/schema.sql` + `db/models.py` — esquema de la DB analítica
  reconstruible (`runs`, `metrics`, `run_factors`).
- `scripts/run_production.py` — entrypoint único para producción.
- `scripts/sync/` — flujo de instituto: `fetch_from_server.sh` →
  `push_to_mlflow.py` → `build_analytics_db.py`, en ese orden, nunca
  salteado.
- `sandbox/` — prototipado rápido, aislado, solo lectura de la DB.
- `PROTOCOL.md` — documento completo de reglas (importado arriba).

## Estado del proyecto

Esqueleto inicial. `run_production.py` y `push_to_mlflow.py` tienen
TODOs marcados donde falta lógica de dominio (BCI). No hay todavía un
segundo dominio (imágenes, PINNs) — no generalizar el `Dockerfile` ni
crear `janus-core-base` hasta que exista uno real (ver PROTOCOL.md
sección 10).