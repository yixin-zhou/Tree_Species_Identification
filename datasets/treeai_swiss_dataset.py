import torch
from torch.utils.data import Dataset
from pathlib import Path
import h5py
from natsort import natsorted
import numpy as np
import torch.nn.functional as F
from pyproj import Transformer

MEAN_AERIAL = torch.tensor([150.89, 92.71, 84.85, 80.70]).view(4, 1, 1)
STD_AERIAL = torch.tensor([36.76, 27.62, 22.48, 26.73]).view(4, 1, 1)

MEAN_S2 = torch.tensor([4304.33, 4159.27, 4057.78, 4328.95, 4571.22, 4644.87, 4837.25, 4700.26, 2823.26, 2319.97]).view(10, 1, 1)
STD_S2 = torch.tensor([3538.00, 3324.23, 3270.07, 3250.53, 2897.39, 2754.50, 2821.52, 2625.95, 1731.56, 1549.30]).view(10, 1, 1)


MEAN_S1 = torch.tensor([3.29, -3.68]).view(2, 1, 1)
STD_S1 = torch.tensor([40.11, 40.53]).view(2, 1, 1)


ID_TO_LABEL = {
    3: 0, 6: 1, 9: 2, 12: 3, 13: 4, 24: 5, 
    26: 6, 30: 7, 36: 8, 43: 9, 56: 10, 63: 11
}

class TreeAISwissDataset(Dataset):
    def __init__(self, 
                 folder,
                 split,
                 target_size=(300, 300),
                 s1_unit='db',
                 s2_fill=0,
                 vhm_compute_method='minus',
                 sentinel_timestamp='monthly'
                 ):
        if not Path(folder).is_dir():
            raise FileNotFoundError(f"Dataset folder not found at: {folder}")

        self.folder = folder
        self.split = split
        self.dataset_split_folder = Path(self.folder) / split
        self._hdf5_file_list = natsorted(list(self.dataset_split_folder.glob("*.hdf5")))
        
        self.target_size = target_size
        self.s1_unit = s1_unit
        self.s2_fill = s2_fill
        self.vhm_method = vhm_compute_method
        self.sentinel_timestamp = sentinel_timestamp
        

        self.transformer = Transformer.from_crs("EPSG:2056", "EPSG:4326", always_xy=True)

    def __len__(self):
        return len(self._hdf5_file_list)

    @staticmethod
    def _temporal_composite(monthly_data, timestamp):
        if timestamp == 'annual':
            composite = np.nanmedian(monthly_data, axis=0)
            return np.expand_dims(composite, axis=0)
        elif timestamp == 'quarterly':
            num_quarters = 4
            time_per_quarter = monthly_data.shape[0] // 4
            composites = []
            for q in range(num_quarters):
                data_slice = monthly_data[q*time_per_quarter : (q+1)*time_per_quarter]
                composites.append(np.nanmedian(data_slice, axis=0))
            return np.stack(composites)
        elif timestamp == 'monthly':
            return monthly_data

    def _convert_labels(self, labels_raw, target_h, target_w):
        bboxes = []
        classes = []
        
        raw_bboxes = labels_raw['bbox']
        raw_ids = labels_raw['category_id']

        for i in range(len(raw_ids)):
            label_id = int(raw_ids[i])
            if label_id not in ID_TO_LABEL:
                continue
            
            norm_xc, norm_yc, norm_w, norm_h = raw_bboxes[i]
            
            xc = norm_xc * target_w
            yc = norm_yc * target_h
            w = norm_w * target_w
            h = norm_h * target_h
            
            xmin = xc - w / 2
            ymin = yc - h / 2
            xmax = xc + w / 2
            ymax = yc + h / 2
            
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(target_w, xmax)
            ymax = min(target_h, ymax)
            
            if xmax > xmin and ymax > ymin:
                bboxes.append([xmin, ymin, xmax, ymax])
                classes.append(ID_TO_LABEL[label_id])
                
        return torch.tensor(bboxes, dtype=torch.float32), torch.tensor(classes, dtype=torch.long)

    def __getitem__(self, idx):
        hdf5_filepath = self._hdf5_file_list[idx]

        with h5py.File(hdf5_filepath, 'r') as hf:
            uav_raw = hf['uav_image'][:] 
            s1_raw = hf['sentinel1_ts'][:]
            s2_raw = hf['sentinel2_ts'][:]
            
            if self.vhm_method == 'minus':
                vhm_raw = np.maximum(hf['dsm'][:] - hf['dem'][:], 0)
            else:
                vhm_raw = np.maximum(hf['vhm'][:], 0)
            
            bounds = hf['metadata'].attrs['bounds']
            center_x_2056 = (bounds[0] + bounds[2]) / 2
            center_y_2056 = (bounds[1] + bounds[3]) / 2
            
            lon, lat = self.transformer.transform(center_x_2056, center_y_2056)
            loc_tensor = torch.tensor([lon, lat], dtype=torch.float32)

            labels_raw = hf['labels'][:]
        
        uav = uav_raw.astype(np.float32)

        lower = np.percentile(uav, 2)
        upper = np.percentile(uav, 98)
        uav = (uav - lower) / (upper - lower + 1e-5)
        uav = np.clip(uav, 0, 1) * 255.0 
        uav_tensor = torch.from_numpy(uav).float()

        s2_ts = self._temporal_composite(s2_raw, self.sentinel_timestamp)
        s2_ts = np.nan_to_num(s2_ts, nan=0.0)
        if s2_ts.max() < 2.0: 
             s2_ts = s2_ts * 10000.0
        s2_ts = np.clip(s2_ts, 0, 10000)
        s2_tensor = torch.from_numpy(s2_ts).float()

        s1_ts = self._temporal_composite(s1_raw, self.sentinel_timestamp)
        s1_ts = np.nan_to_num(s1_ts, nan=-30.0)
        if self.s1_unit == 'linear':
            s1_ts = 10 * np.log10(s1_ts + 1e-6)
        
        if s1_ts.shape[1] > 2:
             s1_ts = s1_ts[:, :2, :, :]
        s1_tensor = torch.from_numpy(s1_ts).float()

        uav_norm = (uav_tensor - MEAN_AERIAL) / STD_AERIAL
        s2_norm = (s2_tensor - MEAN_S2) / STD_S2
        s1_norm = (s1_tensor - MEAN_S1) / STD_S1
        
        uav_final = F.interpolate(uav_norm.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False).squeeze(0)
        
        boxes, classes = self._convert_labels(labels_raw, self.target_size[0], self.target_size[1])

        input_data = {
            'image': uav_final,
            's1_ts': s1_norm,
            's2_ts': s2_norm,
            'vhm': torch.from_numpy(vhm_raw).float().unsqueeze(0),
            'loc': loc_tensor
        }
        
        target = {
            'boxes': boxes,
            'labels': classes
        }

        return input_data, target