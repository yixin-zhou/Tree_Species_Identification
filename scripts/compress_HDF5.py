import h5py
from pathlib import Path
from tqdm import tqdm
import rasterio
import numpy as np
import calendar


def load_monthly_sentinel(sentinel_folder):
    months = list(calendar.month_name[1:])
    monthly_data = []
    for month in months:
        tif_filepath = Path(sentinel_folder) / f"{month}.tif"
        with rasterio.open(tif_filepath) as f:
            monthly_data.append(f.read())

    final_array = np.stack(monthly_data, axis=0)
    return final_array


def load_chelsa_data(filepath):
    keys = [f"BIO{i}" for i in range(1, 20)]
    with open(filepath, 'r') as f:
        line = f.readline()
    values_str = line.strip().split()
    values_float = [float(v) for v in values_str]
    bioclim_dict = dict(zip(keys, values_float))
    return bioclim_dict


def load_coco_label(filepath):
    target_dtype = np.dtype([
        ('bbox', 'f4', (4,)),
        ('category_id', 'i4')
    ])

    parsed_annotations = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            class_id = int(parts[0])
            bbox_coords = [float(p) for p in parts[1:]]
            parsed_annotations.append((tuple(bbox_coords), class_id))

    return np.array(parsed_annotations, dtype=target_dtype)


dataset_path = Path("../data/TreeAI_Swiss")
modalities = ['images', 'DEM', 'DSM', 'VHM', 'Sentinel-1', 'Sentinel-2',
            'Satellite_Embedding', 'climate', 'labels', 'masks']

images_folder = dataset_path / "images"

for image in tqdm(list(images_folder.rglob("*.tif")), desc="Compressing data to HDF5 file"):
    # Load UAV image data
    with rasterio.open(image) as src:
        uav_image = src.read()
        crs = src.crs
        bounds = src.bounds

    # Load DEM
    dem_path = str(image).replace('images','DEM')
    with rasterio.open(dem_path) as src:
        dem = np.squeeze(src.read())

    # Load DSM
    dsm_path = str(image).replace('images','DSM')
    with rasterio.open(dsm_path) as src:
        dsm = np.squeeze(src.read())

    # Load VHM
    vhm_path = str(image).replace('images','VHM')
    with rasterio.open(vhm_path) as src:
        vhm = np.squeeze(src.read())

    # Load mask
    mask_path = str(image).replace('images','masks')
    with rasterio.open(mask_path) as src:
        mask = np.squeeze(src.read())

    # Load Google Satellite Embedding v1
    sate_embed_path = str(image).replace('images','Satellite Embedding')
    with rasterio.open(sate_embed_path) as src:
        sate_embed = src.read()

    # Load Sentinel-1 Time Series
    sentinel1_folder = str(image).replace('images','Sentinel-1').replace('.tif', '')
    s1_ts = load_monthly_sentinel(sentinel1_folder)

    # Load Sentinel-2 Time Series
    sentinel2_folder = str(image).replace('images','Sentinel-2').replace('.tif', '')
    s2_ts = load_monthly_sentinel(sentinel2_folder)

    # Next move to metadata, first CHELSA climate data
    climate_path = str(image).replace('images','climate').replace('.tif', '.txt')
    bioclim = load_chelsa_data(climate_path)

    # Load COCO label
    label_path = str(image).replace('images','labels').replace('.tif', '.txt')
    label_data = load_coco_label(label_path)

    # Compress all modality of data into one HDF5 file
    hdf5_filepath = str(image).replace('images','HDF5').replace('.tif', '.hdf5')
    Path(hdf5_filepath).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_filepath, "w") as hf:
        hf.create_dataset('uav_image', data=uav_image, compression='gzip')
        hf.create_dataset('dem', data=dem, compression='gzip')
        hf.create_dataset('dsm', data=dsm, compression='gzip')
        hf.create_dataset('vhm', data=vhm, compression='gzip')
        hf.create_dataset('mask', data=mask, compression='gzip')
        hf.create_dataset('satellite_embedding', data=sate_embed, compression='gzip')
        hf.create_dataset('sentinel1_ts', data=s1_ts, compression='gzip')
        hf.create_dataset('sentinel2_ts', data=s2_ts, compression='gzip')
        hf.create_dataset('labels', data=label_data, compression='gzip')

        metadata_group = hf.create_group('metadata')

        for key, value in bioclim.items():
            metadata_group.attrs[key] = value
        metadata_group.attrs['crs_wkt'] = crs.to_wkt()
        metadata_group.attrs['bounds'] = list(bounds)

