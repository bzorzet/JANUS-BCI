from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat

from src.utils.paths import PATHS
from .base import BaseEEGDataset
from .subject import Subject


class Lee2019(BaseEEGDataset):
    """
    Motor Imagery dataset from Lee et al. 2019 (GigaScience, DOI 10.5524/100542).
    """
    def __init__(self, sessions = ["session_1", "session_2"], subjects=None, data_to_load = None):
        # 1. Define specific configuration for Lee2019
        path = str(PATHS.data_root / 'MNE-gigadb-data' / 'gigadb-datasets'
                    / 'live' / 'pub' / '10.5524' / '100001_101000' / '100542')

        subjects = list(range(1, 55)) if subjects is None else subjects

        event_id = {'left_hand': 1, 'right_hand': 2}

        # 2. Initialize Base Class
        super().__init__(dataset_path=path, subject_list=subjects, event_id=event_id, code='Lee2019')

        # 3. Database-specific attributes
        self.sfreq = 1000
        self.standard_montage = "standard_1005"

        eeg_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
                'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
                'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10', 'FC3', 'FC4',
                'C5', 'C1', 'C2', 'C6', 'CP3', 'CPz', 'CP4', 'P1', 'P2', 'POz', 'FT9', 'FTT9h',
                'TTP7h', 'TP7', 'TPP9h', 'FT10', 'FTT10h', 'TPP8h', 'TP8', 'TPP10h',
                'F9', 'F10', 'AF7', 'AF3', 'AF4', 'AF8', 'PO3', 'PO4']
        emg_ch = ['EMG1', 'EMG2', 'EMG3', 'EMG4']

        self.ch_names = eeg_ch + emg_ch + ['stim']
        self.ch_types = ["eeg"] * len(eeg_ch) + ["emg"] * len(emg_ch) + ['stim']

        # En el futuro se pueden agregar los otros data_availables acá.
        self.data_availables = ['motor_imagery']

        self.data_to_load = data_to_load if data_to_load is not None else self.data_availables
        self.sessions = sessions
        self.unit_factor = 1e-6 # Convert from uV to V
        self.subjects_metadata = self._load_subjects_metadata()

    def set_data_to_load(self, data_to_load):
        self.data_to_load = data_to_load

    def _load_subjects_metadata(self, path=None):
        if path is None:
            path = PATHS.data_root / 'MNE-gigadb-data' / 'gigadb-datasets' \
                / 'live' / 'pub' / '10.5524' / '100001_101000' / '100542'
        csv_path = Path(path) / 'database_information.csv'
        if csv_path.exists():
            try:
                subjects_metadata = pd.read_csv(csv_path)
            except Exception as e:
                print(f"Error loading subjects metadata: {e}")
                subjects_metadata = None
        else:
            subjects_metadata = None
        return subjects_metadata

    def download(self):
        raise NotImplementedError(
            "TODO: descarga de GigaDB (DOI 10.5524/100542) — pendiente de confirmar "
            f"fuente/mirror. Verificá que los datos ya estén en {self.dataset_path} "
            "o descargalos manualmente (ver https://doi.org/10.5524/100542)."
        )

    def get_subject(self, subject_id: int, session: str = "session_1") -> Subject:
        # Mismo patrón de manejo de path/errores que Cho2017/Dreyer2023Base:
        # si falta el archivo, avisa y devuelve None en vez de romper el barrido.
        session_num = int(session.split('_')[-1])
        file_path = str(Path(self.dataset_path) / f"session{session_num}" / f"s{subject_id}"
                         / f"sess{session_num:02d}_subj{subject_id:02d}_EEG_MI.mat")
        if not Path(file_path).exists():
            print(f"Warning: File not found for subject {subject_id}")
            return None

        mat = loadmat(file_path)
        subject_data = {}
        for data in self.data_to_load:
            subject_data[data] = {}
            if data == "motor_imagery":
                raw_train = self._create_raw_task(mat["EEG_MI_train"][0, 0])
                subject_data[data]['run_1'] = raw_train
        # Metadatos siempre en la raíz
        subject_data['extra_info'] = self._obtain_extra_info(subject_id)
        return Subject(subject_id=subject_id, subject_dict=subject_data)

    def _obtain_extra_info(self, subject_id):
        # Metadatos base (frecuencia, canales, montaje)
        extra_info = {
            'sfreq': self.sfreq,
            'ch_names': self.ch_names,
            'ch_types': self.ch_types,
            'montage_name': self.standard_montage,
            'event_id': self.event_id,
            'unit_factor': self.unit_factor
        }
        # Añadir info del CSV (Age, Gender, etc.)
        if self.subjects_metadata is not None:
            sub_row = self.subjects_metadata[self.subjects_metadata['subject_id'] == subject_id]
            if not sub_row.empty:
                extra_info['personal_metadata'] = sub_row.to_dict(orient='records')[0]
        return extra_info

    def _create_raw_task(self, data_struct):
        """
        Convierte la estructura de MATLAB en un Raw de MNE siguiendo
        el estándar de concatenación NumPy.
        """
        # Extraer matriz de datos (muestras x canales) y transponer a (canales, muestras)
        data_array = data_struct["x"].T
        data_array_emg = data_struct["EMG"].T
        # 1. Escalar los datos de EEG (uV a V)
        scaled_data = data_array * self.unit_factor
        scaled_emg_data = data_array_emg * self.unit_factor

        # 2. Crear una fila de ceros para el canal 'stim'
        stim_row = np.zeros((1, scaled_data.shape[1]))

        # 2.5. Introducir los eventos en el canal 'stim' de forma segura
        # np.atleast_1d previene errores si loadmat carga un solo valor como escalar
        triggers = np.atleast_1d(data_struct["t"]).flatten()
        labels = np.atleast_1d(data_struct["y_dec"]).flatten()

        for trigger, label in zip(triggers, labels):

            sample_idx = int(trigger)

            # Check de seguridad para evitar out-of-bounds
            if 0 <= sample_idx < stim_row.shape[1]:
                stim_row[0, sample_idx] = label

        # 3. Apilar las matrices para obtener (n_canales_eeg + 1, n_times)
        full_data = np.vstack([scaled_data, scaled_emg_data, stim_row])

        # 4. Crear Info usando las listas maestras de la clase
        info = mne.create_info(
            ch_names=self.ch_names,
            ch_types=self.ch_types,
            sfreq=self.sfreq
        )

        # 5. Crear objeto Raw
        raw = mne.io.RawArray(full_data, info=info, verbose=False)

        # 6. Aplicar montaje (MNE ignora canales no-EEG como 'stim')
        try:
            montage = mne.channels.make_standard_montage(self.standard_montage)
            raw.set_montage(montage)
        except Exception as e:
            print(f"Aviso: No se pudo aplicar el montaje estándar. {e}")

        return raw
