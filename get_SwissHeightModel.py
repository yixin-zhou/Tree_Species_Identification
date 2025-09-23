import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
from pathlib import Path
from tqdm import tqdm
import logging
import os
import pandas as pd
import math
from Utils.utils import download_file, clip, mosaic_clip


def locateSwissHeightModel(input_tif):
    with rasterio.open(input_tif) as src:
        epsg_code = src.crs.to_epsg()
        assert epsg_code == 2056, "The CRS of input .tif raster should be EPSG:2056"
        xmin, ymin, xmax, ymax = src.bounds
    
    e_min_idx = math.floor(xmin / 1000.0)
    e_max_idx = math.floor(xmax / 1000.0)
    n_min_idx = math.floor(ymin / 1000.0)
    n_max_idx = math.floor(ymax / 1000.0)

    EN_tiles = []
    for e in range(e_min_idx, e_max_idx+1):
        for n in range(n_min_idx, n_max_idx+1):
            EN_tiles.append(f"{e:04d}-{n:04d}")
    
    return EN_tiles


height_model_links = {'DEM':"ch.swisstopo.swissalti3d-W8cz43uc.csv",
                      'DSM':"ch.swisstopo.swisssurface3d-raster-y1lbk9PU.csv"}


images_folder = Path("data/TreeAI_Swiss/images")
uav_images = list(images_folder.rglob("*.tif"))


for height_model in ['DSM','DEM']:

    download_links = pd.read_csv(height_model_links[height_model], header=None)
    download_links = download_links[0].tolist()
    # print(len(uav_images)) # 9437 images

    HM_save_dir = Path(f'data/TreeAI_Swiss/{height_model}_raw')
    HM_save_dir.mkdir(parents=True, exist_ok=True)

    for uav_image in tqdm(uav_images, desc=f"Locating Swiss {height_model} for UAV images"):
        EN_tiles = locateSwissHeightModel(uav_image)
        for EN_tile in EN_tiles:
            target_dem_links = [link for link in download_links if EN_tile in link]
            if len(target_dem_links) > 1:
                logging.info(f"There are several {height_model} products for {uav_image}")
            elif len(target_dem_links) == 0:
                logging.info(f"There are NO {height_model} product for {uav_image}")
            else:
                download_url = target_dem_links[0]
            url_filename = download_url.split("/")[-1]
            url_save_path = HM_save_dir / url_filename
            if not url_save_path.exists():
                download_file(url=download_url, save_path=HM_save_dir)
        
        