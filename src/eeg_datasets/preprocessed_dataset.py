import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import mne

from src.utils.json_utils import load_json_file
from src.utils.paths import PATHS
from .base import BaseEEGDataset
from .subject import Subject

class PreprocessedDataset(BaseEEGDataset):
    def __init__(self, preprocessing_name, db_name,
                 sessions=None,
                 subjects=None,
                 data_to_load=None,
                 channels=None,
                 classes_to_return=None):

        self.preprocessing_name = preprocessing_name
        self.db_name = db_name
        self.classes_to_return = classes_to_return

        self.base_path = str(PATHS.preprocessed_root / preprocessing_name / db_name)

        # 1. Load the Master Manifest
        self.metadata = self._load_metadata()

        # 2. Determine Scope (Sessions & Contexts)
        self.sessions = sessions if sessions else self.metadata.get('sessions_availables', ['session_1'])
        self.available_contexts = self.metadata.get('data_availables', [])
        self.data_to_load = data_to_load if data_to_load else self.available_contexts

        # 3. Filter Subject List
        # We define subjects_to_use here and save it to the instance
        all_available_subjects = self.metadata.get('subject_list', [])
        self.subject_list = subjects if subjects else all_available_subjects

        # 4. Synchronize properties (sfreq, ch_names, etc.)
        self._update_metadata()

        self.selected_channels = None
        if channels:
            self.select_channels(channels)

        # 5. Initialize Base Class
        # Note: we pass self.subject_list which is now filtered THISIS WEIRD WE HAVE TO THINK MORE THE LOGIC
        super().__init__(
            dataset_path=self.base_path,
            subject_list=self.subject_list,
            event_id=self.event_id,
            code=f"{db_name}_{preprocessing_name}",
            subjects_metadata=self.subjects_metadata,
            data_availables=self.available_contexts
        )

    def download(self):
        raise NotImplementedError(
            "PreprocessedDataset lee datos ya preprocesados, no descarga nada. "
            "Corré el barrido de preprocesamiento (EEGDatabasePreprocessor) sobre "
            "el dataset crudo correspondiente."
        )

    def _update_metadata(self):
        """
        Updates instance attributes using the top-level keys from the metadata JSON.
        These keys represent the current (preprocessed) state of the data.
        """
        # We assume the JSON now has these at the top level.
        # If a key is missing at the top level, we could check 'original_properties' as a fallback.
        orig = self.metadata.get('original_properties', {})

        # Top-level properties (The "Ground Truth" of the preprocessed files)
        self.sfreq = self.metadata.get('sfreq', orig.get('sfreq'))
        self.ch_names = self.metadata.get('ch_names', orig.get('ch_names', []))
        self.ch_types = self.metadata.get('ch_types', orig.get('ch_types', []))
        self.event_id = self.metadata.get('event_id', orig.get('event_id', {}))
        self.standard_montage = self.metadata.get('standard_montage', orig.get('standard_montage', 'standard_1005'))

        # Convenience attributes
        self.n_channels = len(self.ch_names)
        self.ch_names = self.ch_names # To keep consistency with Cho2017
        self.ch_types = self.ch_types

        # Handle subject individual metadata (Age, Gender, etc.)
        # This makes it accessible via self.subjects_metadata for the rest of the app
        self.subjects_metadata = None
        sub_meta = self.metadata.get('subjects_individual_metadata', {})
        if sub_meta:
            # We convert it back to a DataFrame so it matches the BaseEEGDataset interface
            self.subjects_metadata = pd.DataFrame.from_dict(sub_meta, orient='index')

    def _load_metadata(self):
        path = os.path.join(self.base_path, 'dataset_description.json')
        if not os.path.exists(path):
            raise FileNotFoundError(f"The dataset_description.json file does not exist in path {path}")
        return load_json_file(path)

    def select_channels(self, channels):
        if all(ch in self.ch_names for ch in channels):
            self.selected_channels = [self.ch_names.index(ch) for ch in channels]
        else:
            missing = [ch for ch in channels if ch not in self.ch_names]
            raise ValueError(f"The following channels are not available in the dataset: {missing}")

    def get_subject(self, subject_id: int, session: str = None) -> Subject:
        current_session = session if session else self.sessions[0]
        sub_folder = os.path.join(self.base_path, current_session, f"subject_{subject_id:02d}")

        # 1. Prepare Metadata for this specific subject
        # Use the logic from _update_metadata for channels
        if self.selected_channels is not None:
            sub_ch_names = [self.ch_names[i] for i in self.selected_channels]
            sub_ch_types = [self.ch_types[i] for i in self.selected_channels]
        else:
            sub_ch_names = self.ch_names
            sub_ch_types = self.ch_types

        # Base extra_info with dataset-level properties
        extra_info = {
            'sfreq': self.sfreq,
            'event_id': self.event_id,
            'ch_names': sub_ch_names,
            'ch_types': sub_ch_types,
            'standard_montage': self.standard_montage,
            'n_channels': len(sub_ch_names)
        }

        # Add individual subject info (Age, Sex, etc.) from our DataFrame if it exists
        if self.subjects_metadata is not None and subject_id in self.subjects_metadata['subject_id'].values:
            individual_stats = self.subjects_metadata[self.subjects_metadata['subject_id'] == subject_id].to_dict('records')[0]
            extra_info.update(individual_stats)

        if not os.path.exists(sub_folder):
            return Subject(subject_id, {'extra_info': extra_info})

        # 2. Parse Files
        subject_dict = {'extra_info': extra_info}
        prefix_to_remove = f"subject_{subject_id:02d}_"
        files = sorted(os.listdir(sub_folder))
        for file in files:
            if not file.endswith(('.fif', '.npy', '.npz')) or "ica" in file.lower():
                continue

            file_path = os.path.join(sub_folder, file)
            clean_name = file.replace(prefix_to_remove, "").rsplit('.', 1)[0]

            # Match context
            found_context = None
            sorted_contexts = sorted(self.available_contexts, key=len, reverse=True)
            for ctx in sorted_contexts:
                if clean_name.startswith(ctx):
                    found_context = ctx
                    break

            if not found_context:
                continue

            # Parse remainder: context_specifier_type
            remainder = clean_name[len(found_context):].lstrip('_')
            if file.endswith('.npz'):
                # Un .npz combina data+labels en un solo archivo, sin
                # sufijo de tipo en el nombre (a diferencia de los .npy
                # sueltos) — todo el remainder es el specifier.
                data_type = None
                specifier = remainder if remainder else "default"
            else:
                parts = remainder.split('_')
                data_type = parts[-1] # data, labels, raw, epo
                specifier = "_".join(parts[:-1]) if len(parts) > 1 else "default"

            # Build nested dict
            if found_context not in subject_dict:
                subject_dict[found_context] = {}
            if specifier not in subject_dict[found_context]:
                subject_dict[found_context][specifier] = {}
            # This logic is useful, when the file to read are raw or epochs, the information of labels are inside the object.
            # But when the file are npy, we have two files, one for data and other for labels, so we need to load both files to have the complete information of the subject.
            if data_type in ['data', 'labels']: # The two files are needed to have the complete information of the subject, so we save them in the same level of the dict
                subject_dict[found_context][specifier][data_type] = self._load_file(file_path)
            else:
                subject_dict[found_context][specifier]= self._load_file(file_path)

        return Subject(subject_id, subject_dict)

    def _load_file(self, path):
        """Cargador con filtrado de canales integrado."""
        if path.endswith('.npy'):
            data = np.load(path)
            # Si es data (3D: trials, channels, samples) y hay canales seleccionados
            if self.selected_channels is not None and len(data.shape) == 3:
                data = data[:, self.selected_channels, :]
            return data

        elif path.endswith('.npz'):
            npz = np.load(path)
            data = npz['data']
            if self.selected_channels is not None and len(data.shape) == 3:
                data = data[:, self.selected_channels, :]
            result = {'data': data}
            if 'labels' in npz.files:
                result['labels'] = npz['labels']
            return result

        elif path.endswith('_raw.fif'):
            raw = mne.io.read_raw_fif(path, preload=True, verbose=False)
            if self.selected_channels is not None:
                # Usamos los nombres reales para MNE
                ch_names = [self.ch_names[i] for i in self.selected_channels]
                raw.pick(ch_names)
            return raw

        elif path.endswith('_epo.fif'):
            epochs = mne.read_epochs(path, preload=True, verbose=False)
            if self.selected_channels is not None:
                ch_names = [self.ch_names[i] for i in self.selected_channels]
                epochs.pick(ch_names)
            return epochs
        return path

    def flatten_subject_data(self, subject_id, session=None, context=None, specifier=None):
        """
        Collapses a subject's hierarchical data into flat X, y matrices.

        Args:
            subject_id (int): The ID of the subject to process.
            context (str): List of context to include (e.g., 'motor_imagery').
            specifier (str): Optional list of runs/specs to filter (e.g., 'run_1').

        Returns:
            X (np.ndarray), y (np.ndarray), metadata (pd.DataFrame)
        """
        # 1. Get the Subject object using your existing architecture
        sub = self.get_subject(subject_id, session=session)

        context_name = context if context else sub.contexts[0]
        spec_name = specifier if specifier else "all"

        data_content = sub.get_from_path(f"{context_name}:{spec_name}")

        if data_content is None:
            raise ValueError(f"No data found for {context_name}:{spec_name}")

        if not isinstance(data_content, dict):
            runs_to_process = {spec_name: data_content}
        else:
            runs_to_process = data_content

        X_list, y_list, meta_list = [], [], []

        for run_id, content in runs_to_process.items():
            if 'data' in content and 'labels' in content:
                X_run = content['data']
                y_run = content['labels']

                # Aplicar filtrado de canales si se cargó por fuera de Subject
                if self.selected_channels is not None and X_run.shape[1] != len(self.selected_channels):
                    X_run = X_run[:, self.selected_channels, :]

                X_list.append(X_run)
                y_list.append(y_run)

                # REPARACIÓN DE METADATOS: Aquí conservamos el run_id (specifier)
                temp_meta = pd.DataFrame({
                    'session': [session if session else self.sessions[0]] * len(y_run),
                    'subject': [subject_id] * len(y_run),
                    'context': [context_name] * len(y_run),
                    'specifier': [run_id] * len(y_run), # <-- AQUÍ SE GUARDA EL RUN ORIGINAL
                    'labels': y_run
                })
                meta_list.append(temp_meta)

        if not X_list:
            raise ValueError(f"No valid data/labels found for Subject {subject_id}")

        X = np.concatenate(X_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        metadata = pd.concat(meta_list, ignore_index=True)

        return X, y, metadata

    def flatten_pool_data(self, subject_ids, session=None, context=None, specifier=None):
        """
        Aggregates flattened data from a list of subjects into a single pool.

        Args:
            subject_ids (list of int): List of IDs to include in the pool.
            session (str): The session to pull from.
            context (str): The specific context (e.g., 'motor_imagery').
            specifier (str): The run/specifier (e.g., 'run_1').

        Returns:
            X_pool (np.ndarray), y_pool (np.ndarray), metadata_pool (pd.DataFrame)
        """
        X_pool_list = []
        y_pool_list = []
        meta_pool_list = []

        for sub_id in subject_ids:
            try:
                X_sub, y_sub, meta_sub = self.flatten_subject_data(
                    subject_id=sub_id,
                    session=session,
                    context=context,
                    specifier=specifier
                )

                X_pool_list.append(X_sub)
                y_pool_list.append(y_sub)
                meta_pool_list.append(meta_sub)

            except Exception as e:
                print(f"Warning: Could not process subject {sub_id}. Error: {e}")
                continue

        if not X_pool_list:
            raise ValueError(f"No data found for any of the subjects in the pool: {subject_ids}")

        # Concatenate everything into single structures
        X_pool = np.concatenate(X_pool_list, axis=0)
        y_pool = np.concatenate(y_pool_list, axis=0)
        metadata_pool = pd.concat(meta_pool_list, ignore_index=True)

        return X_pool, y_pool, metadata_pool


class PreprocessingReport:
    """
    Lectura de solo lectura de los CSV que escribe EEGDatabasePreprocessor
    (ver DESIGN.md sección 7) — mismo espíritu que sandbox/db_reader.py: no
    toca señal, no escribe nada, solo lee run_registry.csv/stage_metrics.csv/
    <stage_name>_<detail_key>.csv.
    """
    def __init__(self, preprocessing_name: str, db_name: str):
        self.base_path = PATHS.preprocessed_root / preprocessing_name / db_name

    def run_registry(self) -> pd.DataFrame:
        return pd.read_csv(self.base_path / "run_registry.csv")

    def stage_metrics(self, stage_name: Optional[str] = None, metric_name: Optional[str] = None) -> pd.DataFrame:
        df = pd.read_csv(self.base_path / "stage_metrics.csv")
        if stage_name is not None:
            df = df[df["stage_name"] == stage_name]
        if metric_name is not None:
            df = df[df["metric_name"] == metric_name]
        return df

    def detail_table(self, stage_name: str, detail_key: str) -> pd.DataFrame:
        return pd.read_csv(self.base_path / f"{stage_name}_{detail_key}.csv")
