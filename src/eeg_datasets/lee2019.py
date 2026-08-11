import logging
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import mne
from scipy.io import loadmat

from src.utils.paths import PATHS
from .base import BaseEEGDataset
from .subject import Subject

logger = logging.getLogger(__name__)

# Wasabi S3 mirror used by GigaDB for this dataset (DOI 10.5524/100542).
WASABI_URL = "https://s3.ap-northeast-1.wasabisys.com/gigadb-datasets/live/pub/10.5524/100001_101000/100542/"


class Lee2019(BaseEEGDataset):
    """
    Multi-paradigm BCI dataset from Lee et al. 2019 (OpenBMI, GigaScience).

    This dataset contains EEG and EMG recordings from 54 subjects performing
    three canonical BCI paradigms across two recording sessions on different
    days: motor imagery (MI), event-related potential (ERP) speller, and
    steady-state visually evoked potential (SSVEP).

    References
    ----------
    .. [1] Lee, M.H., Kwon, O.Y., Kim, Y.J., Kim, H.K., Lee, Y.E., Williamson, J.,
        Fazli, S. and Lee, S.W., 2019. EEG dataset and OpenBMI toolbox for
        three BCI paradigms: an investigation into BCI illiteracy.
        GigaScience, 8(5). https://doi.org/10.1093/gigascience/giz002

    Data Structure & Channels
    --------------------------
    Raw data is distributed as MATLAB structures (*.mat), one file per
    paradigm per session, each containing separate train/test structs
    (e.g. EEG_MI_train, EEG_MI_test).
    *   **Sampling Rate:** 1000 Hz
    *   **Channels 1-62:** EEG, 10-20 system, nasion-referenced, grounded at AFz
    *   **Channels 63-66:** EMG (both flexor digitorum profundus muscles)
    *   Exact channel order/names are read from the 'chan' field of each .mat
        struct, not hardcoded -- unlike Cho2017, OpenBMI provides this
        per-recording.

    Note: this loader currently only downloads/loads the MI-related files
    (EEG_MI_train / EEG_MI_test) for both sessions -- ERP, SSVEP and Artifact
    files are not fetched by download() by default, since they are outside
    this project's current use of the dataset.

    Experiment Protocols & Trials
    ------------------------------
    Two sessions per subject, on different days, identical protocol. Order
    within a session: ERP, MI, SSVEP.

    +-------------------+--------------------------+---------------------+-------------------------------------------+
    | Context           | Condition / Class        | Trials (Count)      | Description                                |
    +===================+==========================+======================+=============================================+
    | **Motor Imagery** | Left / Right Hand MI     | 100 train, 100 test | Subject imagines grasping with each hand.  |
    | (MI)              |                          |                      |                                             |
    +-------------------+--------------------------+---------------------+-------------------------------------------+
    | **ERP (speller)** | Target / Non-target      | 1980 train,          | 6x6 row-column speller, random-set +       |
    |                   |                          | 2160 test            | face-stimuli presentation.                 |
    +-------------------+--------------------------+---------------------+-------------------------------------------+
    | **SSVEP**         | 5.45/6.67/8.57/12 Hz     | 100 train, 100 test  | Four flicker frequencies, one per          |
    |                   |                          |                      | direction (down/right/left/up).            |
    +-------------------+--------------------------+---------------------+-------------------------------------------+
    | **Resting state** | Eyes open, no task       | 20 recordings        | Recorded before/after every run.           |
    +-------------------+--------------------------+---------------------+-------------------------------------------+
    | **Noise**         | Eye blink, eye movement  | 5 x 10s recordings   | Intentional artifact recordings, once      |
    | (Artifacts)       | (h/v), jaw clenching,    |                      | per session.                                |
    |                   | arm flexing              |                      |                                             |
    +-------------------+--------------------------+---------------------+-------------------------------------------+

    Metadata Fields
    -----------------
    Each .mat struct includes:
    *   **x:** continuous EEG/EMG signal (data points x channels)
    *   **t:** stimulus onset sample indices for each trial
    *   **fs:** sampling rate (1000 Hz)
    *   **y_dec / y_logic:** class labels, integer and one-hot encoded
    *   **y_class:** class name definitions
    *   **chan:** channel names, in recording order

    Participant metadata (age 24-35, 25 female, handedness, prior BCI
    experience, pre/post-session questionnaires) is available separately per
    subject -- not currently parsed by this loader.
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

        self.emg_ch_names = ['EMG1', 'EMG2', 'EMG3', 'EMG4']
        # EEG channel order/names vary per recording in OpenBMI, so they
        # are read from the 'chan' field of each .mat file in get_subject()
        # instead of being fixed here -- not known until a subject is
        # actually loaded.
        self.ch_names = []
        self.ch_types = []

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
        """Download the per-subject, per-session EEG_MI .mat files (each
        one bundles both EEG_MI_train and EEG_MI_test, both in scope)
        from the GigaDB Wasabi mirror (DOI 10.5524/100542), skipping
        files already present on disk. database_information.csv is a
        hand-curated file and is never downloaded here -- only checked
        for, with a warning if missing, since download() must not fail
        because of it."""
        dest_root = Path(self.dataset_path)
        for session in self.sessions:
            session_num = int(session.split('_')[-1])
            for subject_id in self.subject_list:
                filename = f"sess{session_num:02d}_subj{subject_id:02d}_EEG_MI.mat"
                dest_path = dest_root / f"session{session_num}" / f"s{subject_id}" / filename
                if dest_path.exists() and dest_path.stat().st_size > 1024:
                    logger.info(
                        "Skipping subject %d session %d: %s already exists",
                        subject_id, session_num, filename,
                    )
                    continue

                url = WASABI_URL + f"session{session_num}/s{subject_id}/{filename}"
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                logger.info("Downloading subject %d session %d from %s", subject_id, session_num, url)
                try:
                    urllib.request.urlretrieve(url, dest_path)
                except KeyboardInterrupt:
                    if dest_path.exists():
                        dest_path.unlink()
                    raise
                except Exception as e:
                    logger.error("Failed to download %s: %s", filename, e)
                    if dest_path.exists():
                        dest_path.unlink()

        metadata_path = dest_root / "database_information.csv"
        if not metadata_path.exists():
            logger.warning(
                "database_information.csv not found at %s -- this file is a hand "
                "curation, not downloaded automatically. Copy it there manually.",
                metadata_path,
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
        train_struct = mat["EEG_MI_train"][0, 0]

        # Channel order/names for this specific recording -- see the
        # 'chan'-field note in the class docstring and __init__.
        eeg_ch_names = self._read_channel_names(train_struct)
        self.ch_names = eeg_ch_names + self.emg_ch_names + ['stim']
        self.ch_types = ["eeg"] * len(eeg_ch_names) + ["emg"] * len(self.emg_ch_names) + ['stim']

        subject_data = {}
        for data in self.data_to_load:
            subject_data[data] = {}
            if data == "motor_imagery":
                raw_train = self._create_raw_task(train_struct)
                subject_data[data]['run_1'] = raw_train
        # Metadatos siempre en la raíz
        subject_data['extra_info'] = self._obtain_extra_info(subject_id)
        return Subject(subject_id=subject_id, subject_dict=subject_data)

    @staticmethod
    def _read_channel_names(data_struct) -> list:
        """Read EEG channel names from the 'chan' field of a loaded
        EEG_MI_train/test struct. Defensive against the exact wrapping
        scipy.io.loadmat produces for MATLAB cellstr arrays (same
        double np.atleast_1d pattern already used for 't'/'y_dec' in
        _create_raw_task)."""
        chan_field = np.atleast_1d(data_struct["chan"]).flatten()
        return [str(np.atleast_1d(entry).flatten()[0]) for entry in chan_field]

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
