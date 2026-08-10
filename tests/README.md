# Tests

Placeholder. Cuando migres el código real de BCI, agregá al menos:

- Un test de que `PATHS` falla rápido si `.env` apunta a una carpeta
  que no existe (ya lo hace `pydantic-settings`, pero vale confirmarlo
  con un test explícito).
- Un test de que `build_analytics_db.py` no duplica filas si se corre
  dos veces seguidas sobre el mismo `RESULTS_ROOT`.
- Un test de que `sandbox/db_reader.py` efectivamente falla si se
  intenta escribir (confirma que mode=ro está funcionando).
