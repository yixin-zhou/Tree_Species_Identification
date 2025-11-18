import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
from natsort import natsorted
import numpy as np


class TreeAISwissDataset(Dataset):
    def __init__(self, 
                 folder,
                 split,
                 image_mask=False,
                 image_normalized=True,
                 s1_unit='db',
                 s2_mask=True,
                 s2_fill=0,
                 vhm_compute_method='minus',
                 vhm_normalized=True,
                 sentinel_timestamp='monthly'
                 ):
        if not Path(folder).is_dir():
            raise FileNotFoundError(f"Dataset folder not found at: {folder}")

        self.folder = folder

        ALLOWED_SPLITS = {'train', 'val', 'test'}
        if split not in ALLOWED_SPLITS:
            raise ValueError(
                f"Invalid 'split' value: '{split}'. "
                f"Must be one of {ALLOWED_SPLITS}."
            )
        self.split = split
        self.dataset_split_folder = Path(self.folder) / split

        self.image_mask = image_mask
        self.s2_mask = s2_mask

        ALLOWED_S1_UNIT = {'db', 'linear'}
        if s1_unit not in ALLOWED_S1_UNIT:
            raise ValueError(
                f"Invalid 's1_unit' value: '{s1_unit}'. "
                f"Must be one of {ALLOWED_S1_UNIT}."
            )
        self.s1_unit = s1_unit

        ALLOWED_S2_FILLS = {0, 'interpolate'}
        if s2_fill not in ALLOWED_S2_FILLS:
            raise ValueError(
                f"Invalid 's2_fill' value: '{s2_fill}'. "
                f"Must be 0 (for zero imputation) or 'interpolate' (for interpolation)."
            )
        self.s2_fill = s2_fill

        self._hdf5_file_list = natsorted(list(self.dataset_split_folder.glob("*.hdf5")))

        ALLOWED_VHM_METHOD = {'minus', 'direct'}
        if vhm_compute_method not in ALLOWED_VHM_METHOD:
            raise ValueError(
                f"Invalid 'vhm_compute_method' value: '{vhm_compute_method}'. "
                f"Must be 'minus' or 'direct'."
            )
        self.vhm_method = vhm_compute_method
        self.vhm_normalized = vhm_normalized

        ALLOWED_Sentinel_TIMESTAMP = {'monthly', 'quarterly', 'annual'}
        if sentinel_timestamp not in ALLOWED_Sentinel_TIMESTAMP:
            raise ValueError(
                f"Invalid 'sentinel_timestamp' value: '{sentinel_timestamp}'. "
                f"Must be one of {ALLOWED_Sentinel_TIMESTAMP}"
            )
        self.sentinel_timestamp = sentinel_timestamp
        self.image_normalized = image_normalized

    def __len__(self):
        return len(self._hdf5_file_list)

    @staticmethod
    def _temporal_composite(monthly_data, timestamp):
        if timestamp == 'annual':
            composite = np.nanmedian(monthly_data, axis=0)
            return np.expand_dims(composite, axis=0)

        elif timestamp == 'quarterly':
            T = monthly_data.shape[0]
            time_per_quarter = T // 4
            num_quarters = 4

            C, H, W = monthly_data.shape[1:]
            quarterly_composites = np.zeros((num_quarters, C, H, W), dtype=monthly_data.dtype)

            for q in range(num_quarters):
                start_idx = q * time_per_quarter
                end_idx = (q + 1) * time_per_quarter
                quarter_data = monthly_data[start_idx:end_idx]
                composite = np.nanmedian(quarter_data, axis=0)

                quarterly_composites[q] = composite

            return quarterly_composites

        elif timestamp == 'monthly':
            return monthly_data

    def __getitem__(self, idx):
        hdf5_filepath = self._hdf5_file_list[idx]

        data = {}
        with h5py.File(hdf5_filepath) as hf:
            if self.image_normalized:
                data['image'] = hf['uav_image'][:].astype(np.float32) / 65535.0
                data['image_8bit'] = hf['uav_image_8bit'][:].astype(np.float32) / 255.0
            else:
                data['image'] = hf['uav_image'][:].astype(np.float32)
                data['image_8bit'] = hf['uav_image_8bit'][:].astype(np.float32)

            data['satellite_embedding'] = hf['satellite_embedding'][:]
            data['dsm'] = hf['dsm'][:]
            data['dem'] = hf['dem'][:]
            data['mask'] = hf['mask'][:]
            data['labels'] = hf['labels'][:]

            if self.vhm_method == 'minus':
                data['vhm'] = np.maximum(data['dsm'] - data['dem'], 0)
            elif self.vhm_method == 'direct':
                data['vhm'] = np.maximum(hf['vhm'][:], 0)

            if self.vhm_normalized:
                vhm_min = np.min(data['vhm'])
                vhm_max = np.max(data['vhm'])
                data_range = vhm_max - vhm_min

                if data_range == 0:
                    data['vhm'] = np.zeros_like(data['vhm'], dtype=np.float32)
                else:
                    data['vhm'] = (data['vhm'] - vhm_min) / data_range

            bioclim_data = {}
            for key, value in hf['metadata'].attrs.items():
                if key.startswith('BIO'):
                    bioclim_data[key] = float(value)
            sorted_keys = sorted(bioclim_data.keys(), key=lambda x: int(x[3:]))
            bioclim_values = [bioclim_data[key] for key in sorted_keys]
            data['bioclim'] = np.array(bioclim_values, dtype=np.float32)
            data['CRS'] = hf['metadata'].attrs['crs_wkt']
            data['bounds'] = hf['metadata'].attrs['bounds']

            data['s1_ts'] = self._temporal_composite(monthly_data=hf['sentinel1_ts'][:],
                                                     timestamp=self.sentinel_timestamp)

            data['s2_ts'] = self._temporal_composite(monthly_data=hf['sentinel2_ts'][:],
                                                     timestamp=self.sentinel_timestamp)

            if self.s1_unit == 'linear':
                data['s1_ts'] = 10 ** (0.1 * data['s1_ts'])

        final_tensor = {}
        final_tensor['image'] = torch.from_numpy(data['image'])
        final_tensor['image_8bit'] = torch.from_numpy(data['image_8bit'])
        final_tensor['dem'] = torch.from_numpy(data['dem']).float().unsqueeze(0)
        final_tensor['dsm'] = torch.from_numpy(data['dsm']).float().unsqueeze(0)
        final_tensor['vhm'] = torch.from_numpy(data['vhm']).float().unsqueeze(0)
        final_tensor['satellite_embedding'] = torch.from_numpy(data['satellite_embedding']).float()
        final_tensor['s1_ts'] = torch.from_numpy(data['s1_ts']).float()
        final_tensor['s2_ts'] = torch.from_numpy(data['s2_ts']).float()
        final_tensor['bioclim'] = torch.from_numpy(data['bioclim']).float()
        final_tensor['labels'] = data['labels']

        mask_tensor = torch.from_numpy(data['mask']).long()

        return final_tensor, mask_tensor