#!/usr/bin/env bash
# ============================================================
# JANUS-BCI :: trae resultados nuevos desde un server del instituto
# hacia JANUS_RESULTS_ROOT local.
#
# Solo copia archivos — no toca MLFlow ni la DB analítica. Eso lo
# hacen push_to_mlflow.py y build_analytics_db.py, en ese orden,
# DESPUÉS de correr este script. Ver PROTOCOL.md sección 8.
# ============================================================
set -euo pipefail

: "${INSTITUTE_SERVER_HOST:?Definí INSTITUTE_SERVER_HOST en .env}"
: "${INSTITUTE_SERVER_RESULTS_PATH:?Definí INSTITUTE_SERVER_RESULTS_PATH en .env}"
: "${JANUS_RESULTS_ROOT:?Definí JANUS_RESULTS_ROOT en .env}"

echo "Sincronizando ${INSTITUTE_SERVER_HOST}:${INSTITUTE_SERVER_RESULTS_PATH} -> ${JANUS_RESULTS_ROOT}"

rsync -avz --progress \
    "${INSTITUTE_SERVER_HOST}:${INSTITUTE_SERVER_RESULTS_PATH}/" \
    "${JANUS_RESULTS_ROOT}/"

echo "Listo. Ahora corré: python scripts/sync/push_to_mlflow.py"
