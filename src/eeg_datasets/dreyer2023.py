from pathlib import Path

import numpy as np
import pandas as pd
import mne

from src.utils.paths import PATHS
from .base import BaseEEGDataset
from .subject import Subject

_EVENT_ID = {
    'init_run': 32769,
    'end_run': 32770,
    'start_trial': 768,
    'cross_aparition': 786,
    'acoustic_signal': 33282,
    'left_hand': 769,
    'right_hand': 770,
    'start_feedback': 781,
    'end_trial': 800,
}


class Dreyer2023Base(BaseEEGDataset):
    """
    Lógica común a las variantes A/B/C del dataset de Dreyer et al. 2023.
    Cada subclase finita solo declara `FOLDER_PREFIX`, `DATA_SUBDIR`,
    `METADATA_SUFFIX`, `SUBJECT_RANGE` y `DATA_AVAILABLES` — ver DESIGN.md
    sección 8 (decisión: clase base + subclases finitas).

    References
    ----------
    .. [1] Dreyer, P., Roc, A., Pillette, L. et al. A large EEG database with users'
        profile information for motor imagery brain-computer interface research.
        Sci Data 10, 580 (2023). https://doi.org/10.1038/s41597-023-02445-z

    Data Structure & Channels
    -------------------------
    *   **Sampling Rate:** 512 Hz
    *   **EEG (27 Channels):** Fz, FCz, Cz, CPz, Pz, C1, C3, C5, C2, C4, C6, F4, FC2,
        FC4, FC6, CP2, CP4, CP6, P4, F3, FC1, FC3, FC5, CP1, CP3, CP5, P3.
    *   **Reference:** Left earlobe.
    *   **Ground:** FPz.
    *   **EMG (2 Channels):** EMGl, EMGr (Wrist extensors/flexors).
    *   **EOG (3 Channels):** Right eye (Vertical: above/below, Horizontal: side).
    """
    FOLDER_PREFIX: str = ""
    DATA_SUBDIR: str = ""
    METADATA_SUFFIX: str = ""
    SUBJECT_RANGE: range = range(0, 0)
    DATA_AVAILABLES: list = ['motor_imagery_adquisition', 'motor_imagery_online', 'rest']

    def __init__(self, sessions=None, subjects=None, data_to_load=None):
        path = str(PATHS.data_root / 'Dreyer2023' / 'Signals' / self.DATA_SUBDIR)
        subjects = list(self.SUBJECT_RANGE) if subjects is None else subjects

        super().__init__(dataset_path=path, subject_list=subjects, event_id=dict(_EVENT_ID),
                          code=f'Dreyer2023{self.FOLDER_PREFIX}')

        self.sfreq = 512
        self.standard_montage = "standard_1020"
        self.eeg_ch = ['Fz', 'FCz', 'Cz', 'CPz', 'Pz', 'C1', 'C3', 'C5',
                       'C2', 'C4', 'C6', 'F4', 'FC2', 'FC4', 'FC6', 'CP2',
                       'CP4', 'CP6', 'P4', 'F3', 'FC1', 'FC3', 'FC5', 'CP1', 'CP3', 'CP5', 'P3']
        self.eog_ch = ["EOG1", "EOG2", "EOG3"]
        self.emg_ch = ["EMGl", "EMGr"]
        self.ch_names = self.eeg_ch + self.eog_ch + self.emg_ch + ['stim']
        self.ch_types = ["eeg"] * len(self.eeg_ch) + ["eog"] * len(self.eog_ch) + ["emg"] * len(self.emg_ch) + ['stim']

        self.sessions = sessions if sessions is not None else ["session_1"]
        self.data_availables = self.DATA_AVAILABLES
        self.data_to_load = data_to_load if data_to_load is not None else self.data_availables
        self.sessions_available = ['session_1']
        self.unit_factor = 1e-6  # Convert from uV to V
        self.subjects_metadata = self._load_subjects_metadata(path=PATHS.data_root / 'Dreyer2023')

    def set_data_to_load(self, data_to_load):
        self.data_to_load = data_to_load

    def _load_subjects_metadata(self, path=None):
        csv_path = Path(path) / f'database_information_{self.METADATA_SUFFIX}.csv'
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
            f"TODO: descarga de Dreyer2023{self.FOLDER_PREFIX} (Sci Data, DOI "
            "10.1038/s41597-023-02445-z) — pendiente de confirmar fuente. Verificá "
            f"que los datos ya estén en {self.dataset_path} o descargalos manualmente."
        )

    def _create_raw_simple(self, file_path):
        # 1. Read the gdf file
        raw = mne.io.read_raw_gdf(file_path, eog=None, misc=None,
                                   stim_channel='auto', exclude=(),
                                   include=None, preload=True, verbose=None)

        # 2. Create the stim channel
        events_original, events_id_map = mne.events_from_annotations(raw)
        id_to_original_code = {v: int(k) for k, v in events_id_map.items() if k.isdigit()}
        stim_data = np.zeros((1, len(raw)))
        samples = events_original[:, 0]

        internal_ids = events_original[:, 2]
        original_codes = [id_to_original_code.get(idx, 0) for idx in internal_ids]
        # This is only to avoid possibles issues:
        corrected_samples = np.clip(samples, 0, len(raw) - 1)
        stim_data[0, corrected_samples] = original_codes

        info_stim = mne.create_info(
            ch_names=["stim"],
            sfreq=raw.info['sfreq'],
            ch_types=["stim"]
        )

        # Importante mantener first_samp para evitar desfases
        stim_raw = mne.io.RawArray(stim_data, info_stim, first_samp=raw.first_samp)
        raw.add_channels([stim_raw], force_update_info=True)

        # 3. Rename EMG channels
        mapping_names = {'EMGg': 'EMGl', 'EMGd': 'EMGr'}
        raw.rename_channels(mapping_names)

        # 4. Set types of channels
        mapping_types = {ch: 'eeg' for ch in self.eeg_ch}
        mapping_types.update({ch: 'eog' for ch in self.eog_ch})
        mapping_types.update({ch: 'emg' for ch in self.emg_ch})
        mapping_types['stim'] = 'stim'
        raw.set_channel_types(mapping_types)

        # 5. Reorder channels
        raw.reorder_channels(self.ch_names)

        # 6. Set the system
        raw.set_montage(self.standard_montage)
        return raw

    def _obtain_extra_info(self, subject_id=None):
        extra_info = {}
        extra_info['n_imagery_trials_per_run'] = 40

        # Añadir info del CSV (Age, Gender, etc.)
        if self.subjects_metadata is not None and subject_id is not None:
            sub_row = self.subjects_metadata[self.subjects_metadata['subject_id'] == subject_id]
            if not sub_row.empty:
                extra_info['personal_metadata'] = sub_row.to_dict(orient='records')[0]
        return extra_info

    def get_subject(self, subject_id: int, session: str = "session_1") -> Subject:
        subject_path = str(Path(self.dataset_path) / f"{self.FOLDER_PREFIX}{subject_id}")
        if not Path(subject_path).exists():
            print(f"Warning: File not found for subject {subject_id}")
            return None
        subject_data = {}
        prefix = self.FOLDER_PREFIX

        for data in self.data_to_load:
            subject_data[data] = {}

            if data == 'motor_imagery_adquisition':
                for run in [1, 2]:
                    file_path = str(Path(subject_path) / f"{prefix}{subject_id}_R{run}_acquisition.gdf")
                    if Path(file_path).exists():
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
            elif data == 'motor_imagery_online':
                for run in [3, 4, 5, 6]:
                    file_path = str(Path(subject_path) / f"{prefix}{subject_id}_R{run}_onlineT.gdf")
                    if Path(file_path).exists():
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
            elif data == 'motor_imagery':
                for run in [1, 2]:
                    file_path = str(Path(subject_path) / f"{prefix}{subject_id}_R{run}_acquisition.gdf")
                    if Path(file_path).exists():
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
                for run in [3, 4, 5, 6]:
                    file_path = str(Path(subject_path) / f"{prefix}{subject_id}_R{run}_onlineT.gdf")
                    if Path(file_path).exists():
                        subject_data[data][f"run_{run}"] = self._create_raw_simple(file_path)
            elif data == 'rest':
                for baseline in ["CE", "OE"]:
                    file_path = str(Path(subject_path) / f"{prefix}{subject_id}_{baseline}_baseline.gdf")
                    if Path(file_path).exists():
                        subject_data[data][baseline] = self._create_raw_simple(file_path)

        # La info extra va en la raíz
        subject_data['extra_info'] = self._obtain_extra_info(subject_id=subject_id)
        return Subject(subject_id=subject_id, subject_dict=subject_data)


class Dreyer2023A(Dreyer2023Base):
    """Dreyer2023 variante A — 60 sujetos, incluye el contexto combinado 'motor_imagery'."""
    FOLDER_PREFIX = "A"
    DATA_SUBDIR = "DATA A"
    METADATA_SUFFIX = "A"
    SUBJECT_RANGE = range(1, 61)
    DATA_AVAILABLES = ['motor_imagery_adquisition', 'motor_imagery_online', 'rest', 'motor_imagery']


class Dreyer2023B(Dreyer2023Base):
    """Dreyer2023 variante B — 20 sujetos.

    Antes tenía su propio `get_subject` con un bug: buscaba
    `s{id:02d}.mat` (convención de Cho2017) en vez de la carpeta
    `B{id}` con archivos `.gdf` que realmente usa (igual que A/C). Al
    unificar bajo `Dreyer2023Base.get_subject`, que usa `FOLDER_PREFIX`,
    el bug desaparece.
    """
    FOLDER_PREFIX = "B"
    DATA_SUBDIR = "DATA B"
    METADATA_SUFFIX = "B"
    SUBJECT_RANGE = range(61, 81)
    DATA_AVAILABLES = ['motor_imagery_adquisition', 'motor_imagery_online', 'rest', 'motor_imagery']


class Dreyer2023C(Dreyer2023Base):
    """Dreyer2023 variante C — 5 sujetos."""
    FOLDER_PREFIX = "C"
    DATA_SUBDIR = "DATA C"
    METADATA_SUFFIX = "C"
    SUBJECT_RANGE = range(82, 87)
    DATA_AVAILABLES = ['motor_imagery_adquisition', 'motor_imagery_online', 'rest', 'motor_imagery']
