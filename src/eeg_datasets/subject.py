# src/eeg_datasets/subject.py
import mne
import numpy as np

class Subject:
    """
    Data Wrapper for individual subjects for a SPECIFIC SESSION.
    Internal hierarchy: context -> specifics -> Raw
    """
    def __init__(self, subject_id: int, subject_dict: dict):
        self.id = subject_id

        # 1. Separamos la metadata
        self.extra_info = subject_dict.pop('extra_info', {})

        # 2. El diccionario ahora entra directo en los contextos: {'motor_imagery': {...}, 'rest': {...}}
        # (Asumiendo que tu base de datos ya filtró la sesión antes de instanciar el Subject)
        self._data = subject_dict

    # ==========================================
    # NAVEGACIÓN BÁSICA
    # ==========================================

    def _validate_metadata(self, data_array):
        """
        Ensures that the number of channels in the data matches the metadata.
        Returns synchronized ch_names and ch_types.
        """
        # Get data channel count: (n_epochs, n_channels, n_times) or (n_channels, n_times)
        n_channels_in_data = data_array.shape[1] if data_array.ndim == 3 else data_array.shape[0]

        ch_names = self.extra_info.get('ch_names', [])
        ch_types = self.extra_info.get('ch_types', [])
        sfreq = self.extra_info.get('sfreq', 250.0)

        # Logic 1: Exact match
        if len(ch_names) == n_channels_in_data:
            return ch_names, ch_types, sfreq

        # Logic 2: Mismatch - provide generic names to avoid crashing
        print(f"Warning Subject {self.id}: Metadata has {len(ch_names)} channels, "
              f"but data has {n_channels_in_data}. Generating generic names.")

        new_names = [f"CH{i:03d}" for i in range(n_channels_in_data)]
        new_types = ['eeg'] * n_channels_in_data

        return new_names, new_types, sfreq

    def get_metadata(self, key=None):
        """
        Retorna un metadato específico si se provee una 'key'.
        Si no se provee 'key', retorna el diccionario completo de extra_info.
        """
        if key is None:
            return self.extra_info

        return self.extra_info.get(key, None)

    @property
    def contexts(self) -> list:
        return list(self._data.keys())

    def get_specifics(self, context: str) -> dict:
        """Retorna el diccionario de específicas (ej: {'run_1': Raw, 'run_2': Raw})"""
        return self._data.get(context, {})

    # def get_from_path(self, path: str):
    #     """
    #     Resuelve un string de configuración JSON en un objeto mne.io.Raw.
    #     Ya no necesita el argumento 'session'.
    #     """
    #     if ":" not in path:
    #         raise ValueError(f"El path '{path}' debe seguir el formato 'context:specific'.")

    #     context, specific = path.split(':', 1)

    #     if context == "extra_info":
    #         return self.get_metadata(specific)

    #     specifics_dict = self.get_specifics(context)

    #     if not specifics_dict:
    #         return None

    #     # Lógica especial para 'all'
    #     if specific.lower() in ['all']:
    #         return specifics_dict # Retornamos {run_1: data, run_2: data, ...}

    #     # Búsqueda normal
    #     return specifics_dict.get(specific, None)

    def get_from_path(self, path: str):
        """
        Resolves the configuration string.
        Logic:
        - No suffix -> Returns dictionary for ML.
        - :epoch / :epochs -> Converts dict to mne.Epochs.
        - :raw -> Converts dict to mne.io.Raw.
        - If data is ALREADY an MNE object, returns it directly to save computation.
        """
        parts = path.split(':')
        if len(parts) < 2:
            raise ValueError(f"Path '{path}' must follow 'context:specific' or 'context:specific:format'.")

        context, specific = parts[0], parts[1]

        # Safely grab the format, default to 'dict' if nothing is specified
        return_format = parts[2].lower() if len(parts) > 2 else 'dict'

        if context == "extra_info":
            return self.get_metadata(specific)

        specifics_dict = self.get_specifics(context)
        if not specifics_dict:
            return None
        # Here we
        # --- Handle 'all' concatenation ---
        if specific.lower() == 'all':
            data_list = list(specifics_dict.values())
            if not data_list:
                return []

            first_item = data_list[0]

            # If they are already MNE objects, just concatenate them
            if isinstance(first_item, mne.BaseEpochs):
                return mne.concatenate_epochs(data_list)
            elif isinstance(first_item, mne.io.BaseRaw):
                return mne.concatenate_raws(data_list)

            # If they are dicts, check the suffix!
            elif isinstance(first_item, dict):
                if return_format in ['epoch', 'epochs']:
                    return mne.concatenate_epochs([self._to_mne_epochs(d) for d in data_list])
                elif return_format == 'raw':
                    return mne.concatenate_raws([self._to_mne_raw(d) for d in data_list])

                # If no suffix (or 'dict'), return the un-concatenated dictionary of runs
                return specifics_dict

        # --- Normal single-run retrieval ---
        data = specifics_dict.get(specific, None)
        if data is None:
            return None

        # 1. If it's already an MNE object, DO NOTHING. Return as-is.
        if isinstance(data, (mne.BaseEpochs, mne.io.BaseRaw)):
            return data

        # 2. If it's a Dictionary, STRICTLY follow the suffix
        if isinstance(data, dict):
            if return_format in ['epoch', 'epochs']:
                return self._to_mne_epochs(data)
            elif return_format == 'raw':
                return self._to_mne_raw(data)
            else:
                return data # Default ML dict format -> {'X': ..., 'y': ...}

        return data

    def iter_all_runs(self):
        """Generador limpio sin el nivel de sesión."""
        for context_name, specifics in self._data.items():
            for specific_name, raw_obj in specifics.items():
                yield context_name, specific_name, raw_obj

    def __repr__(self):
        return f"<Subject {self.id} | Contexts: {self.contexts}>"


    # ==========================================
    # SMART CONVERTERS (Type-Aware)
    # ==========================================

    def _to_mne_epochs(self, data_dict: dict) -> mne.EpochsArray:
        """Transforma el diccionario de datos a un mne.EpochsArray"""
        # Aseguramos que data_dict sea un diccionario con 'X' e 'y'
        if not isinstance(data_dict, dict) or 'data' not in data_dict or 'labels' not in data_dict:
            raise ValueError("Para convertir a Epochs, los datos deben ser un dict con claves 'X' e 'y'.")

        X = data_dict['data']
        y = data_dict['labels']

        ch_names, ch_types, sfreq = self._validate_metadata(X)

        info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types=ch_types)

        # Crear la matriz de eventos requerida por MNE: shape (n_epochs, 3)
        # Formato MNE: [sample_index, previous_event_id (usually 0), current_event_id]
        events = np.zeros((X.shape[0], 3), dtype=int)
        events[:, 0] = np.arange(X.shape[0]) * X.shape[2] # Timestamps sintéticos
        events[:, 2] = y.astype(int)

        # Mapeo de IDs (ej: {'motor_imagery': 1, 'rest': 2})
        default_event_id = {str(int(lbl)): int(lbl) for lbl in np.unique(y)}
        event_id_map = self.extra_info.get('event_id', default_event_id)

        tmin = self.extra_info.get('tmin', 0.0)

        epochs = mne.EpochsArray(
            data=X,
            info=info,
            events=events,
            event_id=event_id_map,
            tmin=tmin
        )
        return epochs

    def _to_mne_raw(self, data_dict: dict) -> mne.io.RawArray:
        """
        Transforms data into a continuous RawArray.
        Checks if a stim channel already exists to avoid duplicates.
        """
        X = data_dict['data']
        # Labels are usually passed separately in the dict for ML purposes
        y = data_dict.get('labels', np.zeros(X.shape[0])).astype(int)

        # 1. Get base metadata
        ch_names, ch_types, sfreq = self._validate_metadata(X)

        # 2. Flatten 3D (epochs) to 2D (continuous)
        if X.ndim == 3:
            n_eps, n_ch, n_t = X.shape
            X_cont = X.transpose(1, 0, 2).reshape(n_ch, n_eps * n_t)
        else:
            X_cont = X
            n_eps, n_t = 1, X.shape[1]

        # 3. Determine if we need to add a stim channel
        if 'stim' in ch_names:
            # Stim already exists, just find its index
            stim_idx = ch_names.index('stim')
            full_data = X_cont.copy()
            # We update the existing stim channel with our labels 'y'
            if X.ndim == 3:
                for i in range(n_eps):
                    full_data[stim_idx, i * n_t] = y[i]
            else:
                full_data[stim_idx, 0] = y[0] if y.size > 0 else 0

            full_names = ch_names
            full_types = ch_types
        else:
            # Stim does NOT exist, we stack a new row
            stim_channel = np.zeros((1, X_cont.shape[1]))
            if X.ndim == 3:
                for i in range(n_eps):
                    stim_channel[0, i * n_t] = y[i]
            else:
                stim_channel[0, 0] = y[0] if y.size > 0 else 0

            full_data = np.vstack([X_cont, stim_channel])
            full_names = ch_names + ['stim']
            full_types = ch_types + ['stim']

        # 4. Create MNE object
        info = mne.create_info(ch_names=full_names, sfreq=sfreq, ch_types=full_types)
        raw = mne.io.RawArray(full_data, info, verbose=False)

        # 5. Handle Montage (if EEG channels are present)
        if 'standard_montage' in self.extra_info:
            try:
                raw.set_montage(self.extra_info['standard_montage'])
            except Exception:
                pass # Non-EEG channels are ignored by set_montage automatically

        # 6. Add Boundaries for filtering
        if X.ndim == 3:
            onsets = np.arange(1, n_eps) * (n_t / sfreq)
            annots = mne.Annotations(onset=onsets, duration=[0.01]*len(onsets),
                                     description=["BAD_boundary"]*len(onsets))
            raw.set_annotations(annots)

        return raw
