import rasterio
from pathlib import Path
from rasterio.windows import from_bounds
import geopandas as gpd
from tqdm import tqdm
import numpy as np


def getAcquireYear(grid_id, anno_df):
    samples = anno_df[(anno_df['grid_id'] == grid_id)]
    years = samples['Year'].unique()
    if len(years) == 1:
        return years[0]
    else:
        print(f"There are several acquiring years for this grid {grid_id}, including {years}")
        return np.max(years)


VHM_SIZE = 50

vhm_models = {
              2022:"https://os.zhdk.cloud.switch.ch/envicloud/doi/1000001.1/2022/landesforstinventar-vegetationshoehenmodell_stereo_2022_2056.tif",
              2021:"https://os.zhdk.cloud.switch.ch/envicloud/doi/1000001.1/2021/landesforstinventar-vegetationshoehenmodell_stereo_2021_2056.tif",
              2020:"https://os.zhdk.cloud.switch.ch/envicloud/doi/1000001.1/2020/landesforstinventar-vegetationshoehenmodell_stereo_2020_2056.tif",
              2019:"https://os.zhdk.cloud.switch.ch/envicloud/doi/1000001.1/2019/landesforstinventar-vegetationshoehenmodell_stereo_2019_2056.tif",
              2018:"https://os.zhdk.cloud.switch.ch/envicloud/doi/1000001.1/2019/landesforstinventar-vegetationshoehenmodell_stereo_2019_2056.tif",
              2017:"https://os.zhdk.cloud.switch.ch/envicloud/doi/1000001.1/2016/landesforstinventar-vegetationshoehenmodell_stereo_2016_2056.tif",
              }

ANNOTATIONS_SHP = "../data/shapefile/grid/swiss_tree_annotations_with_filtered_grid.shp"
images_folder = Path("../data/TreeAI_Swiss_60/masks")
vhm_save_dir = Path("../data/TreeAI_Swiss_60/VHM")

uav_images = list(images_folder.rglob("*.tif"))
anno_df = gpd.read_file(ANNOTATIONS_SHP)

for uav_image in tqdm(uav_images, desc="Clipping VHM for each grid"):
    acq_year = getAcquireYear(int(uav_image.stem), anno_df)

    with rasterio.open(uav_image) as src:
        epsg_code = src.crs.to_epsg()
        assert epsg_code == 2056, "The CRS of input .tif raster should be EPSG:2056"
        xmin, ymin, xmax, ymax = src.bounds

    vhm_url = vhm_models[acq_year]
    with rasterio.open(vhm_url) as vhm:
        assert vhm.crs.to_epsg() == 2056, "The CRS of VHM Model should be EPSG:2056"
        win = from_bounds(left=xmin, bottom=ymin, right=xmax, top=ymax, transform=vhm.transform)

        vhm_save_dir = Path(str(uav_image).replace('masks','VHM'))
        vhm_save_dir.parent.mkdir(parents=True, exist_ok=True)

        data = vhm.read(window=win, boundless=True)
        transform = vhm.window_transform(win)

        meta = vhm.meta.copy()
        meta.update(dict(
            driver="GTiff",
            height=data.shape[-2],
            width=data.shape[-1],
            transform=transform,
            count=data.shape[0],
            dtype=data.dtype.name,
            tiled=True
        ))

        with rasterio.open(vhm_save_dir, "w", **meta) as dst:
            dst.write(data)
