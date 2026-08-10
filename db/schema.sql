-- ============================================================
-- JANUS-BCI :: esquema de la DB analítica (janus_analytics.db)
--
-- Esta base es 100% RECONSTRUIBLE: build_analytics_db.py la borra y
-- la vuelve a poblar escaneando RESULTS_ROOT. Nunca es la fuente de
-- verdad — los CSVs en disco lo son. Nunca se edita a mano.
-- ============================================================

CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,     -- coincide con el run_id de MLFlow
    project_name        TEXT NOT NULL,
    strategy_name        TEXT NOT NULL,
    recipe_name          TEXT NOT NULL,
    script_type          TEXT NOT NULL,         -- train_dl | analysis | preprocessing
    context_source        TEXT,                 -- dataset / fuente de señal
    context_partition      TEXT,                 -- ej. subject_08
    context_replicate       TEXT,                 -- ej. split_8_model_399
    status               TEXT,                  -- pending | running | success | failed
    git_commit_hash        TEXT,
    docker_image          TEXT,
    config_path           TEXT,
    timestamp_start        TEXT,
    timestamp_end          TEXT
);

-- Métricas en formato largo: no hace falta saber de antemano qué se va
-- a medir. Cada fila = una métrica de un run puntual.
CREATE TABLE IF NOT EXISTS metrics (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id      TEXT NOT NULL REFERENCES runs(run_id),
    split       TEXT,               -- train | val | test | NULL si no aplica
    metric_name TEXT NOT NULL,
    value       REAL NOT NULL
);

-- Factores del contexto en formato largo: permite agregar seed, fold,
-- source_dataset, target_dataset, etc. sin tocar el esquema cuando
-- aparece una estrategia nueva.
CREATE TABLE IF NOT EXISTS run_factors (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id       TEXT NOT NULL REFERENCES runs(run_id),
    factor_name  TEXT NOT NULL,     -- fold | seed | init | source_dataset | ...
    factor_value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_metrics_run_id ON metrics(run_id);
CREATE INDEX IF NOT EXISTS idx_metrics_name ON metrics(metric_name);
CREATE INDEX IF NOT EXISTS idx_factors_run_id ON run_factors(run_id);
CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_name, strategy_name, recipe_name);
