import rasterio
from rasterio.windows import from_bounds, transform
from rasterio.merge import merge
from pathlib import Path
from tqdm import tqdm
import logging
import os
import pandas as pd
import math
from Utils.utils import download_file, clip, mosaic_clip

DOWNLOAD_RAW = True
H, W = 120, 120


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("swiss_height_model.log"), logging.StreamHandler()]
)


def locateSwissHeightModel(input_tif):
    with rasterio.open(input_tif) as src:
        crs = src.crs
        epsg_code = crs.to_epsg()

        if epsg_code == None:
            if "CH1903+" in crs.to_string() and "LV95" in crs.to_string():
                epsg_code = 2056
        assert epsg_code == 2056, "The CRS of input .tif raster should be EPSG:2056"

        xmin, ymin, xmax, ymax = src.bounds

        clip_window = from_bounds(*src.bounds, transform=src.transform)

    e_min_idx = math.floor(xmin / 1000.0)
    e_max_idx = math.floor(xmax / 1000.0)
    n_min_idx = math.floor(ymin / 1000.0)
    n_max_idx = math.floor(ymax / 1000.0)

    EN_tiles = []
    for e in range(e_min_idx, e_max_idx + 1):
        for n in range(n_min_idx, n_max_idx + 1):
            EN_tiles.append(f"{e:04d}-{n:04d}")

    return EN_tiles, clip_window


if __name__ == '__main__':
    height_model_links = {'DEM': "../data/Height_Model_Download_links/ch.swisstopo.swissalti3d-W8cz43uc.csv",
                          'DSM': "../data/Height_Model_Download_links/ch.swisstopo.swisssurface3d-raster-y1lbk9PU.csv"}

    images_folder = Path("../data/TreeAI_Swiss_60/masks")
    uav_images = list(images_folder.rglob("*.tif"))

    for height_model in ['DSM', 'DEM']:

        try:
            download_links = pd.read_csv(height_model_links[height_model], header=None)[0].tolist()
        except Exception as e:
            logging.error(f"Failed to read {height_model} CSV file: {e}")
            continue
        # print(len(uav_images)) # 9437 images

        HM_save_dir = Path(f'../data/TreeAI_Swiss_60/{height_model}_raw')
        HM_save_dir.mkdir(parents=True, exist_ok=True)

        for uav_image in tqdm(uav_images, desc=f"Locating Swiss {height_model} for UAV images"):
            try:
                EN_tiles, clip_window = locateSwissHeightModel(uav_image)
            except Exception as e:
                logging.error(f"Failed to locate tiles for {uav_image}: {e}")
                continue
            # h, w = int(clip_window.height), int(clip_window.width)
            # assert h == H and w == W, "H and W should be 500"

            EN_tiles_filename = []
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
                    if not url_save_path.exists() and DOWNLOAD_RAW == True:
                        try:
                            download_file(url=download_url, save_path=HM_save_dir)
                        except Exception as e:
                            logging.error(f"Download failed for {url_filename}: {e}")
                            continue
                    EN_tiles_filename.append(os.path.join(str(HM_save_dir), url_filename))

            clipped_hm_save_dir = Path(str(uav_image).replace('masks', height_model))
            clipped_hm_save_dir.parent.mkdir(exist_ok=True, parents=True)

            if clipped_hm_save_dir.exists():
                continue

            try:
                if len(EN_tiles_filename) == 1:
                    clip(
                        input_raster=EN_tiles_filename[0],
                        clip_raster=uav_image,
                        output=clipped_hm_save_dir
                    )
                else:
                    mosaic_clip(
                        input_rasters=EN_tiles_filename,
                        clip_raster=uav_image,
                        output=clipped_hm_save_dir
                    )
            except Exception as e:
                logging.error(f"Failed to process {uav_image.name}: {e}")
                continue