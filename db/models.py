"""
Modelos SQLAlchemy que reflejan db/schema.sql.

`build_analytics_db.py` es la ÚNICA pieza de código con permiso real de
escritura acá. Todo lo demás — en particular cualquier script bajo
sandbox/ — tiene que usar get_readonly_session() o db_reader.query().
"""
from __future__ import annotations

from sqlalchemy import Column, Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

from src.utils.paths import PATHS

Base = declarative_base()


class Run(Base):
    __tablename__ = "runs"

    run_id = Column(String, primary_key=True)
    project_name = Column(String, nullable=False)
    strategy_name = Column(String, nullable=False)
    recipe_name = Column(String, nullable=False)
    script_type = Column(String, nullable=False)
    context_source = Column(String)
    context_partition = Column(String)
    context_replicate = Column(String)
    status = Column(String)
    git_commit_hash = Column(String)
    docker_image = Column(String)
    config_path = Column(String)
    timestamp_start = Column(String)
    timestamp_end = Column(String)

    metrics = relationship("Metric", back_populates="run")
    factors = relationship("RunFactor", back_populates="run")


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    split = Column(String)
    metric_name = Column(String, nullable=False)
    value = Column(Float, nullable=False)

    run = relationship("Run", back_populates="metrics")


class RunFactor(Base):
    __tablename__ = "run_factors"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(String, ForeignKey("runs.run_id"), nullable=False)
    factor_name = Column(String, nullable=False)
    factor_value = Column(String, nullable=False)

    run = relationship("Run", back_populates="factors")


def _engine_url(readonly: bool = False) -> str:
    db_path = PATHS.analytics_db
    if readonly:
        # SQLite URI mode=ro: cualquier INSERT/UPDATE/DELETE falla con
        # OperationalError en vez de escribir silenciosamente. Esto es
        # lo que hace cumplir "sandbox lee, nunca escribe" a nivel
        # técnico, no solo de convención.
        return f"sqlite:///file:{db_path}?mode=ro&uri=true"
    return f"sqlite:///{db_path}"


def get_engine(readonly: bool = False):
    return create_engine(_engine_url(readonly))


def get_session(readonly: bool = False):
    engine = get_engine(readonly=readonly)
    return sessionmaker(bind=engine)()


def get_readonly_session():
    """Único punto de entrada permitido para sandbox/ y análisis exploratorio."""
    return get_session(readonly=True)
