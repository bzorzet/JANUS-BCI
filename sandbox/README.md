# Sandbox

Ver PROTOCOL.md sección 1 para el contexto completo. Reglas concretas:

1. **Nada acá se trackea.** No hay `script_progress.csv`, no hay
   `metrics_results.csv` con contrato, no hay push a MLFlow. Es
   territorio libre para probar una idea rápido.
2. **Lectura sí, escritura no.** Un script de sandbox puede leer la DB
   analítica central (`db/janus_analytics.db`) para comparar contra
   resultados de producción — pero solo con `query()` de
   `sandbox/db_reader.py`. Cualquier intento de escritura falla a
   propósito (conexión SQLite en modo `mode=ro`).
3. **Una carpeta por proyecto.** `sandbox/<project_name>/` — adentro,
   estructura libre (notebooks, scripts sueltos, CSVs de prueba).
4. **Promoción manual, nunca automática.** Si algo de acá funciona y
   merece quedar trackeado, se reescribe como un script "real" bajo
   `src/` que siga el contrato de `script_progress.csv` /
   `metrics_results.csv` y corra vía `scripts/run_production.py`.
