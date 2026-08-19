class BCIDataProviderAdapter:
    """Envuelve un dataset BCI (PreprocessedDataset u otro con la misma
    interfaz) para exponer el contrato DataProvider genérico. Es EL ÚNICO
    lugar donde 'session'/'subject' aparecen fuera de eeg_datasets/."""

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