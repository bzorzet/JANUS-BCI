"""
`DataProvider` -- contrato genérico, agnóstico de dominio, para obtener
datos de una o varias "unidades" (lo que sea que eso signifique en cada
dominio: sujeto en BCI, paciente, empresa, etc.). Ningún `TrainingSchema`
sabe qué es una "unidad" en términos concretos -- solo sabe pedirle datos
a un `DataProvider` por identificador.

`BCIDataProviderAdapter` es EL ÚNICO lugar de todo `src/training/` donde
aparece vocabulario específico de EEG/BCI ("subject", "session") -- envuelve
un dataset BCI (PreprocessedDataset u otro con la misma interfaz de
flatten_subject_data/flatten_pool_data) para exponer el contrato genérico
que el resto del framework espera. Si mañana se usa este framework para
otro dominio, este es el único archivo que se reemplaza por un adaptador
análogo -- nada de execution/ ni splitting/ necesita cambiar.
"""
from typing import Any, List, Protocol, Tuple


class DataProvider(Protocol):
    """Contrato mínimo que cualquier fuente de datos debe cumplir para
    poder usarse con los TrainingSchema de src/training/."""

    def get_unit(self, unit_id: Any, **kwargs) -> Tuple[Any, Any, Any]:
        """Devuelve (X, y, metadata) para UNA unidad."""
        ...

    def get_units(self, unit_ids: List[Any], **kwargs) -> Tuple[Any, Any, Any]:
        """Devuelve (X, y, metadata) agregados para VARIAS unidades."""
        ...

    def list_units(self) -> List[Any]:
        """Devuelve la lista de identificadores de unidad disponibles."""
        ...


class BCIDataProviderAdapter:
    """Adapta un dataset BCI (con flatten_subject_data/flatten_pool_data,
    ej. PreprocessedDataset) al contrato DataProvider genérico.

    `default_session` se usa salvo que el caller pase `session` explícito
    en kwargs (ej. para CrossSessionSchema a futuro, que necesitaría variar
    la sesión entre train y test)."""

    def __init__(self, dataset, default_session: str):
        self.dataset = dataset
        self.default_session = default_session

    def get_unit(self, unit_id, **kwargs):
        session = kwargs.get("session", self.default_session)
        return self.dataset.flatten_subject_data(unit_id, session=session)

    def get_units(self, unit_ids, **kwargs):
        session = kwargs.get("session", self.default_session)
        return self.dataset.flatten_pool_data(unit_ids, session=session)

    def list_units(self):
        return self.dataset.subject_list