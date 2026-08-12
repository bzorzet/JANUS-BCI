import torch
import numpy as np
from typing import Union

# Create a simple torch dataset
class MIBCI_SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, classification = True):
        # Check if X is a tensor or numpy array
        if isinstance(X,np.ndarray):
            self.X = torch.tensor(X, dtype=torch.float)
        else:
            self.X = X
        if isinstance(y,np.ndarray):
            if classification:
                self.y = torch.tensor(y, dtype=torch.long)
            else:
                self.y = torch.tensor(y, dtype=torch.float)
        else:
            if classification:
                self.y = y.long()
            else:
                self.y = y.float()
        self.n = X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n
    
#%%###########  DATASETS FOR MISSING DATA ##############
class MIBCI_SimpleDataset_MissingData(torch.utils.data.Dataset):

    def __init__(self, X, y, classification=True,
                 originals_channels=None,
                 channels_to_keep=None,
                 missing_data_strategy="zeros",
                 adjacency_matrix_config=None):
        
        self.originals_channels = originals_channels
        self.channels_to_keep = channels_to_keep
        self.missing_data_strategy = missing_data_strategy
        self.adjacency_matrix_config = adjacency_matrix_config
        # If input X has fewer channels than originals_channels, expand it
        if X.shape[1] < len(self.originals_channels):
            # Create expanded tensor with zeros
            expanded_X = torch.zeros((X.shape[0], len(self.originals_channels),X.shape[2]), dtype=torch.float, device=X.device)

            # Map channels_to_keep to their indices in originals_channels
            keep_indices = [self.originals_channels.index(ch) for ch in channels_to_keep]
            # Insert original data into expanded positions
            if isinstance(X, np.ndarray):
                expanded_X[:, keep_indices, :] = torch.tensor(X, dtype=torch.float)
            else:
                expanded_X[:, keep_indices, :] = X
            self.X = expanded_X
        else:
            # Normal case, just convert to tensor if needed
            if isinstance(X, np.ndarray):
                self.X = torch.tensor(X, dtype=torch.float)
            else:
                self.X = X

        # Process y
        if isinstance(y, np.ndarray):
            if classification:
                self.y = torch.tensor(y, dtype=torch.long)
            else:
                self.y = torch.tensor(y, dtype=torch.float)
        else:
            if classification:
                self.y = y.long()
            else:
                self.y = y.float()

        self.n = self.X.shape[0]
        self.transform_data()

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def __len__(self):
        return self.n

    def transform_data(self):
        # Find missing channels = originals_channels not in channels_to_keep
        missing_channels = [ch for ch in self.originals_channels if ch not in self.channels_to_keep]
        missing_indices = [self.originals_channels.index(ch) for ch in missing_channels]

        # Mask for missing channels
        mask = torch.ones(self.X.shape[1], dtype=torch.bool)
        mask[missing_indices] = False

        if self.missing_data_strategy == "zeros":
            self.X[:, missing_indices] = 0.
        elif self.missing_data_strategy == "mean":
            mean_values = self.X[:, mask].mean(dim=1, keepdim=True)
            self.X[:, missing_indices] = mean_values
        elif self.missing_data_strategy == "neighbourhood_interpolation":
            # Weighted neighbor interpolation using adjacency matrix and einsum
            # Normalize adjacency rows (avoid division by zero)
            A_norm =  self.A  / ( self.A .sum(dim=1, keepdim=True) + 1e-8)
            # Only update missing channels
            X_interp = torch.einsum('ij,bjt->bit', A_norm, self.X)
            self.X[:, missing_indices, :] = X_interp[:, missing_indices, :]




class FrequencyMaskingDataset(torch.utils.data.Dataset):

    """
    Dataset that masks time series data based on a specified frequency band.
    Can perform either bandpass or band-reject filtering.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, fs: int,
                 freqcut: Union[list[float], float],
                 mask_type: str = 'bandpass', order: int = 8,
                 keep_power: bool = True):
        """
        Args:
            X (np.ndarray): The feature data (trials, channels, time).
            y (np.ndarray): The labels.
            fs (int): The sampling rate of the data.
            lowcut (float): The lower cutoff frequency for the band.
            highcut (float): The upper cutoff frequency for the band.
            mask_type (str): 'bandpass' to keep the band, 'bandstop' to remove it.
            order (int): The order of the Butterworth filter.
        """
        self.X = X
        self.y = y
        self.n_samples = self.X.shape[0]
        self.fs = fs
        self.freqcut = freqcut
        self.mask_type = mask_type
        self.order = order
        self.keep_power = keep_power
        self._transform_data()
        self.y = torch.tensor(self.y , dtype=torch.long) if isinstance(self.y ,np.ndarray) else self.y .long()
        self.X = torch.tensor(self.X, dtype=torch.float) if isinstance(self.X,np.ndarray) else self.X.float()


    def _transform_data(self):
        from scipy.signal import butter, filtfilt
        """
        Applies a bandpass or band-reject filter to the dataset.
        """
        nyq = 0.5 * self.fs

        freqcut = np.array(self.freqcut) / nyq

        b, a = butter(self.order, freqcut, btype=self.mask_type)

        # Apply the filter with zero-phase distortion
        X_ = self.X.cpu().numpy() if isinstance(self.X, torch.Tensor) else self.X
        filtered_data = filtfilt(b, a, X_)
        filtered_data = np.ascontiguousarray(filtered_data)
        if self.keep_power:
            power = (X_**2).sum(axis=-1)
            power_filtered = (filtered_data**2).sum(axis=-1)
            k = np.sqrt(power / power_filtered)
            k = k[:, :, np.newaxis]
            filtered_data = k * filtered_data
        self.X = torch.tensor(filtered_data, dtype=torch.float).to(self.X.device)


    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


class MultiFrequencyBandMaskingDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, fs: int,
                 filters: dict, keep_power: bool = True): # Note: Order 8 suggested
        self.X = X
        self.y = y
        self.n_samples = self.X.shape[0]
        self.fs = fs
        self.filters = filters
        self.keep_power = keep_power
        self._transform_data()
        self.y = torch.tensor(self.y , dtype=torch.long) if isinstance(self.y ,np.ndarray) else self.y .long()
        self.X = torch.tensor(self.X, dtype=torch.float) if isinstance(self.X,np.ndarray) else self.X.float()


    def _transform_data(self):
        from scipy.signal import butter, sosfiltfilt
        import numpy as np
        import torch

        # 1. Convert to numpy
        data_to_process = self.X.cpu().numpy() if isinstance(self.X, torch.Tensor) else self.X
        # Make a copy so we don't overwrite the original unexpectedly
        processed_data = data_to_process.copy()

        # 2. Get filter configurations
        filters_name = list(self.filters.keys())

        # 3. SEQUENTIAL FILTERING (The "Chain" approach)
        # Instead of summing, each filter acts on the result of the previous one
        for key in filters_name:
            config = self.filters[key]
            freqcut = config.get('freqcut')
            order = config.get('order')
            mask_type = config.get('mask_type')

            # Use SOS for stability and precision
            sos = butter(order, freqcut, btype=mask_type, fs=self.fs, output='sos')

            # Apply to the SAME data variable (cascading)
            processed_data = sosfiltfilt(sos, processed_data, axis=-1)

        # 4. Power Normalization (Optional but kept as per your logic)
        processed_data = np.ascontiguousarray(processed_data)
        if self.keep_power:
            original_total_power = (data_to_process**2).sum(axis=-1).mean()
            processed_total_power = (processed_data**2).sum(axis=-1).mean()

            if processed_total_power > 1e-10:
                scaling_factor = np.sqrt(original_total_power / processed_total_power)
                processed_data *= scaling_factor

        # 5. Back to Torch
        self.X = torch.tensor(processed_data, dtype=torch.float).to(self.X.device) if isinstance(self.X, torch.Tensor) else torch.tensor(processed_data, dtype=torch.float)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, index: int):
        return self.X[index], self.y[index]


# class MIBCI_SpatialGraphDataset(torch.utils.data.Dataset):
#     def __init__(self, X, y, config_adjacency_matrix = None,
#                  name_adjacency_matrix = None,
#                  name_electrode_order = None,
#                  channels_to_keep = None):
#         # Check if X is a tensor or numpy array
#         if isinstance(X,np.ndarray):
#             self.X = torch.tensor(X, dtype=torch.float)
#         else:
#             self.X = X
#         if isinstance(y,np.ndarray):
#             self.y = torch.tensor(y, dtype=torch.long)
#         else:
#             self.y = y.long()
#         device = X.device
#         self.n = X.shape[0]
#         self.A = self.load_adjacency_matrix(config_adjacency_matrix,
#                                             name_adjacency_matrix,
#                                             name_electrode_order,
#                                             channels_to_keep).to(device)


#     def load_adjacency_matrix(self,config_adjacency_matrix=None,
#                             name_adjacency_matrix = None,
#                             name_electrode_order = None,
#                             channels_to_keep = None):
#         import os
#         from global_config import DIRECTORY_TO_SAVE_ROOT
#         from src.utils.json_utils import load_json_file
#         path_to_load = os.path.join(DIRECTORY_TO_SAVE_ROOT,
#                                     "adjacency_matrix",
#                                     config_adjacency_matrix)
#         # Load adjacency matrix npy file
#         adjacency_matrix = np.load(os.path.join(path_to_load,name_adjacency_matrix))
#         # Load electrodes order json file
#         electrodes_order = load_json_file(os.path.join(path_to_load,name_electrode_order))
#         # Reorganize adjacency matrix and electrodes order
#         if channels_to_keep:
#             reorganized_channels = [channel for channel in electrodes_order if channel in channels_to_keep]
#             reorganized_indices = [electrodes_order.index(channel) for channel in reorganized_channels]
#             adjacency_matrix = adjacency_matrix[reorganized_indices][:, reorganized_indices]

#         return torch.tensor(adjacency_matrix, dtype=torch.float)

#     def __getitem__(self, idx):
#         return {"X": self.X[idx], "A":self.A }, self.y[idx]

#     def __len__(self):
#         return self.n

#     def get_data(self, idx = None):
#         if idx is None:
#             return  (self.X, self.A) , self.y
#         else:
#             return  (self.X[idx], self.A ), self.y[idx]

# class MIBCI_GraphDataset(torch.utils.data.Dataset):
#     def __init__(self, X, y, config_adjacency_matrix = None,
#                  name_adjacency_matrix = None,
#                  name_electrode_order = None,
#                  channels_to_keep = None):
#         # Check if X is a tensor or numpy array
#         if isinstance(X,np.ndarray):
#             self.X = torch.tensor(X, dtype=torch.float)
#         else:
#             self.X = X
#         if isinstance(y,np.ndarray):
#             self.y = torch.tensor(y, dtype=torch.long)
#         else:
#             self.y = y.long()
#         self.n = X.shape[0]
#         self.edges_weight = None
#         self.edges_index = None
#         if name_adjacency_matrix is not None and channels_to_keep is not None:
#             self.load_edges(config_adjacency_matrix,
#                             name_adjacency_matrix,
#                             name_electrode_order,
#                             channels_to_keep)

#     def load_edges(self,config_adjacency_matrix=None,
#                     name_adjacency_matrix = None,
#                     name_electrode_order = None,
#                     channels_to_keep = None):
#         import networkx as nx
#         import os
#         from global_config import DIRECTORY_TO_SAVE_ROOT
#         from src.utils.json_utils import load_json_file
#         path_to_load = os.path.join(DIRECTORY_TO_SAVE_ROOT,
#                                     "adjacency_matrix",
#                                     config_adjacency_matrix)
#         # Load adjacency matrix npy file
#         adjacency_matrix = np.load(os.path.join(path_to_load,name_adjacency_matrix))
#         # Load electrodes order json file
#         electrodes_order = load_json_file(os.path.join(path_to_load,name_electrode_order))
#         # Reorganize adjacency matrix and electrodes order
#         if channels_to_keep:
#             reorganized_channels = [channel for channel in electrodes_order if channel in channels_to_keep]
#             reorganized_indices = [electrodes_order.index(channel) for channel in reorganized_channels]
#             adjacency_matrix = adjacency_matrix[reorganized_indices][:, reorganized_indices]

#         G = nx.convert_matrix.from_numpy_array(adjacency_matrix)
#         self.edges_index = torch.tensor(list(G.edges)).t().contiguous()
#         self.edges_weight = torch.tensor(list(nx.get_edge_attributes(G, 'weight').values()))

#     def __getitem__(self, idx):
#         return {"X": self.X[idx], "edge_index":self.edges_index, "edge_weight":self.edges_weight}, self.y[idx]

#     def __len__(self):
#         return self.n

#     def get_data(self, idx = None):
#         if idx is None:
#             return  (self.X, self.edges_index, self.edges_weight) , self.y
#         else:
#             return  (self.X[idx], self.edges_index, self.edges_weight), self.y[idx]




# class MIBCI_SpatialGraphDataset_MissingData(torch.utils.data.Dataset):
#     def __init__(self, X, y, config_adjacency_matrix = None,
#                  name_adjacency_matrix = None,
#                  name_electrode_order = None,
#                  originals_channels = None,
#                  channels_to_keep = None,
#                  missing_data_strategy = "zeros"):

#         self.channels_to_keep = channels_to_keep
#         self.missing_data_strategy = missing_data_strategy
#         self.originals_channels = originals_channels
#         self.config_adjacency_matrix = config_adjacency_matrix
#         self.name_adjacency_matrix = name_adjacency_matrix
#         self.name_electrode_order = name_electrode_order
#         # If input X has fewer channels than originals_channels, expand it
#         if X.shape[1] < len(self.originals_channels):
#             # Create expanded tensor with zeros
#             expanded_X = torch.zeros((X.shape[0], len(self.originals_channels),X.shape[2]), dtype=torch.float, device=X.device)

#             # Map channels_to_keep to their indices in originals_channels
#             keep_indices = [self.originals_channels.index(ch) for ch in channels_to_keep]
#             # Insert original data into expanded positions
#             if isinstance(X, np.ndarray):
#                 expanded_X[:, keep_indices, :] = torch.tensor(X, dtype=torch.float)
#             else:
#                 expanded_X[:, keep_indices, :] = X
#             self.X = expanded_X
#         else:
#             # Normal case, just convert to tensor if needed
#             if isinstance(X, np.ndarray):
#                 self.X = torch.tensor(X, dtype=torch.float)
#             else:
#                 self.X = X

#         if isinstance(y,np.ndarray):
#             self.y = torch.tensor(y, dtype=torch.long)
#         else:
#             self.y = y.long()
#         device = X.device
#         self.n = X.shape[0]
#         self.A = self.load_adjacency_matrix(config_adjacency_matrix,
#                                             name_adjacency_matrix,
#                                             name_electrode_order,
#                                             originals_channels).to(device)

#         self.transform_data()

#     def transform_data(self):
#         # Find missing channels = originals_channels not in channels_to_keep
#         missing_channels = [ch for ch in self.originals_channels if ch not in self.channels_to_keep]
#         missing_indices = [self.originals_channels.index(ch) for ch in missing_channels]

#         # Mask for missing channels
#         mask = torch.ones(self.X.shape[1], dtype=torch.bool)
#         mask[missing_indices] = False

#         if self.missing_data_strategy == "zeros":
#             self.X[:, missing_indices] = 0.
#         elif self.missing_data_strategy == "mean":
#             mean_values = self.X[:, mask].mean(dim=1, keepdim=True)
#             self.X[:, missing_indices] = mean_values
#         elif self.missing_data_strategy == "neighbourhood_interpolation":
#             # Weighted neighbor interpolation using adjacency matrix and einsum
#             # Normalize adjacency rows (avoid division by zero)
#             A_norm =  self.A  / ( self.A .sum(dim=1, keepdim=True) + 1e-8)
#             # Only update missing channels
#             X_interp = torch.einsum('ij,bjt->bit', A_norm, self.X)
#             self.X[:, missing_indices, :] = X_interp[:, missing_indices, :]


#     def load_adjacency_matrix(self,config_adjacency_matrix=None,
#                             name_adjacency_matrix = None,
#                             name_electrode_order = None,
#                             channels_to_keep = None):
#         import os
#         from global_config import DIRECTORY_TO_SAVE_ROOT
#         from src.utils.json_utils import load_json_file
#         path_to_load = os.path.join(DIRECTORY_TO_SAVE_ROOT,
#                                     "adjacency_matrix",
#                                     config_adjacency_matrix)
#         # Load adjacency matrix npy file
#         adjacency_matrix = np.load(os.path.join(path_to_load,name_adjacency_matrix))
#         # Load electrodes order json file
#         self.electrodes_order = load_json_file(os.path.join(path_to_load,name_electrode_order))

#         # Reorganize adjacency matrix and electrodes order
#         if channels_to_keep:
#             reorganized_channels = [channel for channel in self.electrodes_order if channel in channels_to_keep]
#             reorganized_indices = [self.electrodes_order.index(channel) for channel in reorganized_channels]
#             adjacency_matrix = adjacency_matrix[reorganized_indices][:, reorganized_indices]

#         return torch.tensor(adjacency_matrix, dtype=torch.float)

#     def __getitem__(self, idx):
#         return {"X": self.X[idx], "A":self.A }, self.y[idx]

#     def __len__(self):
#         return self.n

#     def get_data(self, idx = None):
#         if idx is None:
#             return  (self.X, self.A) , self.y
#         else:
#             return  (self.X[idx], self.A ), self.y[idx]


# #%%###########  DATASETS FOR ABLATION STUDIES DATA ##############
# class MaskedTimeSeriesDataset(torch.utils.data.Dataset):
#     import pandas as pd
#     """
#     A custom PyTorch Dataset that masks time series data based on a temporal window
#     during initialization for improved efficiency.
#     """
#     def __init__(self, X, y, metadata, inside_window = True):
#         """
#         Args:
#             X (np.ndarray): The feature data, a 3D NumPy array (trials, channels, time).
#             y (np.ndarray): The labels, a 1D NumPy array.
#             metadata (pd.DataFrame): DataFrame with 'idx', 'temporal_centroid', and
#                                      'temporal_std' columns, correctly split and aligned with X and y.
#         """
#         # Store data as PyTorch tensors. We will mask it in the next step.
#         self.y = torch.tensor(y, dtype=torch.long) if isinstance(y,np.ndarray) else y.long()
#         self.X = torch.tensor(X, dtype=torch.float) if isinstance(X,np.ndarray) else X.float()

#         # Store the metadata.
#         self.metadata = metadata
#         self.inside_window = inside_window
#         self.n_samples = len(self.metadata)
#         self.seq_len = self.X.shape[2]

#         # Call the private method to perform the masking process.
#         self._transform_data()

#     def _transform_data(self):
#         """
#         Applies the temporal mask to the entire dataset at once.
#         """
#         for index in range(self.n_samples):
#             # Get the row from the split metadata DataFrame
#             meta_row = self.metadata.iloc[index]

#             # Get the centroid and standard deviation from the metadata row
#             centroid = meta_row['temporal_centroid']
#             std = meta_row['temporal_std']

#             # Define the masking window
#             start_mask_idx = int(np.floor(centroid - std))
#             end_mask_idx = int(np.ceil(centroid + std))

#             # Clamp the indices to ensure they are within the bounds of the time series
#             start_mask_idx = max(0, start_mask_idx)
#             end_mask_idx = min(self.seq_len, end_mask_idx)

#             # Apply the mask directly to the tensor in-place
#             if start_mask_idx >= end_mask_idx:
#                 continue

#             # Check if the masking is done inside or outside the window
#             if self.inside_window:
#                 # Mask inside the window (default behavior)
#                 self.X[index, :, start_mask_idx:end_mask_idx] = 0
#             else:
#                 # Mask outside the window (before and after the window)
#                 if start_mask_idx > 0:
#                     self.X[index, :, :start_mask_idx] = 0
#                 if end_mask_idx < self.seq_len:
#                     self.X[index, :, end_mask_idx:] = 0

#     def __len__(self) -> int:
#         """Returns the total number of samples in the dataset."""
#         return self.n_samples

#     def __getitem__(self, index: int):
#         """
#         Retrieves a single pre-masked sample.

#         Args:
#             index (int): The zero-based index of the sample in the *split* dataset.

#         Returns:
#             tuple: A tuple containing the masked feature tensor and the label tensor.
#         """
#         # Simply return the pre-masked data, as the transformation is already done
#         return self.X[index], self.y[index]
