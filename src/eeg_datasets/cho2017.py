from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat

from src.utils.paths import PATHS
from .base import BaseEEGDataset
from .subject import Subject

class Cho2017(BaseEEGDataset):
    """
    Motor Imagery dataset from Cho et al. 2017 (GigaScience).

    This dataset contains EEG and EMG recordings from 52 subjects performing Left/Right
    hand Motor Imagery (MI) and real movement tasks.

    References
    ----------
    .. [1] Cho, H., Ahn, M., Ahn, S., Kwon, M. and Jun, S.C., 2017.
           EEG datasets for motor imagery brain computer interface.
           GigaScience. https://doi.org/10.1093/gigascience/gix034

    Data Structure & Channels
    -------------------------
    The raw data is loaded from MATLAB structures ('*.mat').
    *   **Sampling Rate:** 512 Hz
    *   **Channels 1-64:** EEG (10-20 System)
    *   **Channels 65-68:** EMG (EMG1, EMG2, EMG3, EMG4)

    Experiment Protocols & Trials
    -----------------------------
    The dataset includes four distinct experimental contexts. The trial structure
    and event timing are detailed below:

    +-------------------+-----------------------+-------------------+-------------------------------------------+
    | Context           | Condition / Class     | Trials (Count)    | Description                               |
    +===================+=======================+===================+===========================================+
    | **Motor Imagery** | Left Hand MI          | 100 or 120        | Subject imagines left hand movement.      |
    | (MI)              | Right Hand MI         | 100 or 120        | Subject imagines right hand movement.     |
    +-------------------+-----------------------+-------------------+-------------------------------------------+
    | **Real Movement** | Left Hand Mov.        | 20                | Subject performs real left hand grasp.    |
    |                   | Right Hand Mov.       | 20                | Subject performs real right hand grasp.   |
    +-------------------+-----------------------+-------------------+-------------------------------------------+
    | **Resting State** | Rest                  | 1 (Continuous)    | Resting state with eyes-open condition.   |
    +-------------------+-----------------------+-------------------+-------------------------------------------+
    | **Noise**         | Eye Blinking          | 2 (5 sec each)    | Intentional eye blinking artifacts.       |
    | (Artifacts)       | Eyeball Up/Down       | 2 (5 sec each)    | Intentional vertical eye movements.       |
    |                   | Eyeball Left/Right    | 2 (5 sec each)    | Intentional horizontal eye movements.     |
    |                   | Jaw Clenching         | 2 (5 sec each)    | Intentional jaw clenching artifacts.      |
    |                   | Head Movement         | 2 (5 sec each)    | Intentional head movement artifacts.      |
    +-------------------+-----------------------+-------------------+-------------------------------------------+

    Metadata Fields
    ---------------
    The `extra_info` dictionary returned by `get_data` contains:
    *   **senloc/psenloc:** 3D sensor locations and spherical projections.
    *   **bad_trial_idx_mi:** Indices of MI trials rejected due to EMG activity.
    *   **bad_trial_idx_voltage:** Indices of MI trials rejected due to voltage > 100uV.
    *   **n_imagery_trials:** Exact count of MI trials for the subject.
    """
    def __init__(self, sessions = ["session_1"], subjects=None, data_to_load = None):
        # 1. Define specific configuration for Cho
        path = str(PATHS.data_root / 'MNE-gigadb-data' / 'gigadb-datasets'
                    / 'live' / 'pub' / '10.5524' / '100001_101000' / '100295' / 'mat_data')

        subjects = list(range(1, 53)) if subjects is None else subjects
        event_id = {'left_hand': 1, 'right_hand': 2}

        # 2. Initialize Base Class
        super().__init__(dataset_path=path, subject_list=subjects, event_id=event_id, code='Cho2017')

        # 3. Database-specific attributes
        self.sfreq = 512
        self.standard_montage = "standard_1005"
        eeg_ch = [
            "Fp1", "AF7", "AF3", "F1", "F3", "F5", "F7", "FT7", "FC5", "FC3", "FC1",
            "C1", "C3", "C5", "T7", "TP7", "CP5", "CP3", "CP1", "P1", "P3", "P5", "P7",
            "P9", "PO7", "PO3", "O1", "Iz", "Oz", "POz", "Pz", "CPz", "Fpz", "Fp2",
            "AF8", "AF4", "AFz", "Fz", "F2", "F4", "F6", "F8", "FT8", "FC6", "FC4",
            "FC2", "FCz", "Cz", "C2", "C4", "C6", "T8", "TP8", "CP6", "CP4", "CP2",
            "P2", "P4", "P6", "P8", "P10", "PO8", "PO4", "O2",
        ]

        emg_ch = ["EMG1", "EMG2", "EMG3", "EMG4"]
        self.ch_names = eeg_ch + emg_ch + ['stim']
        self.ch_types = ["eeg"] * len(eeg_ch) + ["emg"] * len(emg_ch) + ['stim']

        self.data_availables = ['motor_imagery', 'movement', 'rest', 'noise']
        self.data_to_load = data_to_load if data_to_load is not None else self.data_availables
        self.sessions = sessions
        self.unit_factor = 1e-6 # Convert from uV to V
        self.subjects_metadata = self._load_subjects_metadata()

    def set_data_to_load(self, data_to_load):
        self.data_to_load = data_to_load

    def _load_subjects_metadata(self, path=None):
        # Check if the file exists
        if path is None:
            path = PATHS.data_root / 'MNE-gigadb-data' / 'gigadb-datasets' \
                / 'live' / 'pub' / '10.5524' / '100001_101000' / '100295'
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
            "TODO: descarga de GigaDB (DOI 10.5524/100295) — pendiente de confirmar "
            f"fuente/mirror. Verificá que los datos ya estén en {self.dataset_path} "
            "o descargalos manualmente (ver https://doi.org/10.5524/100295)."
        )

    def get_subject(self, subject_id: int, session: str = "session_1") -> Subject:
        file_path = str(Path(self.dataset_path) / f's{subject_id:02d}.mat')
        if not Path(file_path).exists():
            print(f"Warning: File not found for subject {subject_id}")
            return None

        mat = loadmat(file_path, squeeze_me=True, struct_as_record=False)
        eeg_struct = mat['eeg']

        subject_data = {}

        for data in self.data_to_load:
            subject_data[data] = {}
            if data == "noise":
                noises = getattr(eeg_struct, data)
                for i, noise in enumerate(['eye_blink', 'eyeball_movement_ud',
                                            'eyeball_movement_lr', 'jaw_clenching',
                                            'head_movement_lr']):
                    raw = self._create_raw_simple(noises[i])
                    subject_data[data][noise] = raw
            elif data in ['motor_imagery', 'movement']:
                raw = self._create_raw_task(eeg_struct, data)
                subject_data[data]['run_1'] = raw
            else:
                data_array = getattr(eeg_struct, data)
                raw = self._create_raw_simple(data_array)
                subject_data[data]['recording'] = raw

        # Metadatos siempre en la raíz
        subject_data['extra_info'] = self._obtain_extra_info(eeg_struct, subject_id)
        return Subject(subject_id=subject_id, subject_dict=subject_data)

    def _create_raw_simple(self, data_array):
        """
        Creates an MNE Raw object from a NumPy array, adding a stim channel.
        Input data_array shape: (68, times)
        """
        # 1. Scale the EEG/EMG data (uV to V)
        scaled_data = data_array * self.unit_factor

        # 2. Create a row of zeros for the 'stim' channel
        # Shape must be (1, n_times)
        stim_row = np.zeros((1, scaled_data.shape[1]))

        # 3. Stack them together to get (69, n_times)
        full_data = np.vstack([scaled_data, stim_row])

        # 4. Create Info using the master lists
        info = mne.create_info(
            ch_names=self.ch_names,
            ch_types=self.ch_types,
            sfreq=self.sfreq
        )

        # 5. Create Raw object
        raw = mne.io.RawArray(full_data, info=info, verbose=False)

        # 6. Apply montage (MNE ignores non-EEG channels like 'emg' and 'stim' automatically)
        montage = mne.channels.make_standard_montage(self.standard_montage)
        raw.set_montage(montage)

        return raw

    def _create_raw_task(self, eeg_data, context):
        ctx = 'imagery' if context == 'motor_imagery' else 'movement'

        event = getattr(eeg_data, f'{ctx}_event')

        left = np.vstack([getattr(eeg_data, f'{ctx}_left') * self.unit_factor, event * self.event_id['left_hand']])
        right = np.vstack([getattr(eeg_data, f'{ctx}_right') * self.unit_factor, event * self.event_id['right_hand']])
        data_array =  np.hstack([left, np.zeros((left.shape[0], 500)), right])

        montage = mne.channels.make_standard_montage(self.standard_montage)
        info = mne.create_info(ch_names=self.ch_names , ch_types=self.ch_types, sfreq=self.sfreq)
        raw = mne.io.RawArray(data_array , info=info, verbose=False)
        raw.set_montage(montage)

        return raw

    def _obtain_extra_info(self, eeg_data, subject_id):
        # Metadatos base (frecuencia, canales, montaje)
        extra_info = {
            'sfreq': self.sfreq,
            'ch_names': self.ch_names,
            'ch_types': self.ch_types,
            'montage_name': self.standard_montage,
            'event_id': self.event_id,
            'unit_factor': self.unit_factor
        }

        # First we have to obtain the bad trials correlated with EMG artifacts
        left_bad_trial_idx = getattr(getattr(eeg_data, 'bad_trial_indices'),'bad_trial_idx_mi')[0]
        right_bad_trial_idx = getattr(getattr(eeg_data, 'bad_trial_indices'),'bad_trial_idx_mi')[1] + getattr(eeg_data, 'n_imagery_trials') # Offset for right trials
        extra_info['bad_trial_idx_mi'] = np.hstack([left_bad_trial_idx, right_bad_trial_idx])

        # First we have to obtain the bad trials for above 100 uV
        left_bad_trial_idx = getattr(getattr(eeg_data, 'bad_trial_indices'),'bad_trial_idx_voltage')[0]
        right_bad_trial_idx = getattr(getattr(eeg_data, 'bad_trial_indices'),'bad_trial_idx_voltage')[1] + getattr(eeg_data, 'n_imagery_trials') # Offset for right trials
        extra_info['bad_trial_idx_voltage'] = np.hstack([left_bad_trial_idx, right_bad_trial_idx])

        extra_info['senloc'] = getattr(eeg_data, 'senloc')
        extra_info['psenloc'] = getattr(eeg_data, 'psenloc')

        extra_info['n_imagery_trials'] = getattr(eeg_data, 'n_imagery_trials')
        extra_info['n_movement_trials'] = getattr(eeg_data, 'n_movement_trials')

        # Añadir info del CSV (Age, Gender, etc.)
        if self.subjects_metadata is not None:
            # Buscamos la fila del sujeto (asumiendo que hay una columna 'subject' o similar)
            sub_row = self.subjects_metadata[self.subjects_metadata['subject_id'] == subject_id]
            if not sub_row.empty:
                extra_info['personal_metadata'] = sub_row.to_dict(orient='records')[0]

        return extra_info
