"""
Único punto de entrada permitido para leer la DB analítica desde
sandbox/. Abre la conexión en modo estrictamente lectura: cualquier
INSERT/UPDATE/DELETE falla con OperationalError en vez de corromper
(o simplemente ensuciar) la base reconstruible.

Uso típico dentro de sandbox/<tu_proyecto>/algo.py:

    from sandbox.db_reader import query
    df = query("SELECT * FROM metrics WHERE metric_name = 'accuracy'")
"""
import pandas as pd

from db.models import get_engine


def query(sql: str) -> pd.DataFrame:
    engine = get_engine(readonly=True)
    return pd.read_sql(sql, engine)
