import os
from abc import ABC, abstractmethod

from .subject import Subject


class BaseEEGDataset(ABC):
    """
    Abstract Base Class for all EEG Datasets.
    Enforces a standard interface for loading data with specific contexts.
    """
    def __init__(self, dataset_path, subject_list, event_id=None, code=None, subjects_metadata=None, data_availables=None):
        """
        Parameters:
        -----------
        dataset_path : str
            Absolute path to the dataset folder.
        subject_list : list of int
            List of valid subject IDs.
        event_id : dict
            Mapping of event names to integer codes (e.g., {'left_hand': 1}).
        code : str
            Unique identifier for the dataset (e.g., 'Cho2017').
        """
        self.dataset_path = dataset_path
        self.subject_list = subject_list
        self.event_id = event_id if event_id is not None else {}
        self.code = code if code is not None else self.__class__.__name__
        self.subjects_metadata = subjects_metadata
        self.data_availables = data_availables
        # Verify path exists
        if not os.path.exists(self.dataset_path):
            print(f"WARNING: Dataset path does not exist: {self.dataset_path}")

    def _validate_inputs(self, subjects, data_to_load):
        """
        Helper to standardize inputs.
        """
        if subjects is None:
            subjects = self.subject_list

        # Ensure subjects is a list
        if not isinstance(subjects, list):
            if isinstance(subjects, (int, float)):
                subjects = [int(subjects)]
            else:
                subjects = list(subjects)

        # Ensure data_to_load is a list
        if data_to_load is None:
            raise ValueError("Data to load cannot be None. Please specify (e.g., ['MI', 'rest']).")
        if isinstance(data_to_load, str):
            data_to_load = [data_to_load]

        return subjects, data_to_load

    @abstractmethod
    def download(self):
        """
        Descarga el dataset si no está presente en dataset_path.
        """
        pass

    @abstractmethod
    def get_subject(self, subject_id: int, session: str = "session_1") -> Subject:
        """
        Este es el único método que las clases hijas (como Cho2017)
        DEBEN implementar. Lee el archivo y devuelve el objeto Subject.
        """
        pass

    def get_subjects(self, subjects: list = None, session: str = "session_1") -> list:
        """
        Método de conveniencia. Si le das una lista de IDs, te devuelve
        una lista de objetos Subject usando el método anterior.
        """
        ids = subjects if subjects is not None else self.subject_list
        if isinstance(ids, int):
            ids = [ids]

        return [self.get_subject(s_id, session) for s_id in ids]

    def get_subject_metadata(self):
        """Returns the pandas DataFrame with subject info."""
        if hasattr(self, 'subjects_metadata'):
            return self.subjects_metadata
        elif hasattr(self, 'metadata'):  # Fallback if you named it 'metadata'
            return self.metadata
        return None

    def get_subjects_list(self):
        return self.subject_list

    def get_data_available(self):
        return self.data_availables

    def _load_subjects_metadata(self):
        pass
