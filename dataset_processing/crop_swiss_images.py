import geopandas as gpd
from pathlib import Path
from Utils.utils import mosaic_clip, add_overvier2GeoTiff
import logging


def path_win2linux(win_path, linux_remote_dir):
    path_no_drive = win_path.split(':', 1)[-1]
    path_unix = path_no_drive.replace('\\', '/')
    linux_path = linux_remote_dir.rstrip('/') + path_unix
    return linux_path


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("clip_log.txt", mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


LINUX = True
REMOTE_DIR = "/run/user/1000/gvfs/smb-share:server=ites-formdata.ethz.ch,share=treedetection/"
IMAGE_SIZE = 600
ANNOTATIONS_SHP = "data/shapefile/grid/swiss_tree_annotations_with_filtered_grid.shp"
FISHNET_SHP = "data/shapefile/grid/swiss_tree_fishnet_60m.shp"
DATASET_DIR = "../data/TreeAI_Swiss_60/"

out_masks_dir = Path(DATASET_DIR) / "masks"
out_images_dir = Path(DATASET_DIR) / "images"
out_images_dir.mkdir(parents=True, exist_ok=True)

anno_df = gpd.read_file(ANNOTATIONS_SHP)
fishnet_df = gpd.read_file(FISHNET_SHP)

grid_id_list = anno_df['grid_id'].unique()

for grid_id in grid_id_list:
    try:
        logger.info(f"Begin clipping for grid {int(grid_id)}")

        samples_within_grid = anno_df[(anno_df['grid_id'] == grid_id)]
        grid_polygon = fishnet_df[(fishnet_df['grid_id'] == grid_id)].iloc[0].geometry
        images_url = samples_within_grid['ImageURI'].unique()

        grid_tif = out_masks_dir / f"{int(grid_id):04d}.tif"
        image_save_path = out_images_dir / f"{int(grid_id):04d}.tif"

        if len(images_url) == 1:
            if LINUX:
                image_url = path_win2linux(images_url[0], REMOTE_DIR)
            else:
                image_url = images_url[0].replace("Y", "Z")
            fully_contained = mosaic_clip(input_rasters=[image_url], clip_raster=grid_tif, output=image_save_path)
            # if not fully_contained:
            #     print("Only one corresponding image URI, but not fully contained.")
            #     not_fully_contained_one.append(int(grid_id))

        else:
            if LINUX:
                images_url = [path_win2linux(url, REMOTE_DIR) for url in images_url]
            else:
                images_url = list(images_url)
                images_url = [url.replace('Y', 'Z') for url in images_url]
            fully_contained = mosaic_clip(input_rasters=images_url, clip_raster=grid_tif, output=image_save_path)
            add_overvier2GeoTiff(image_save_path)
            # if not fully_contained:
            #     print("Several corresponding images URI, but not fully contained.")
            #     not_fully_contained_several.append(int(grid_id))
        add_overvier2GeoTiff(image_save_path)
        logger.info(f"Successfully clip and generate overview for grid {grid_id}")
        logger.info("=========================================================================")
    except Exception as e:
        logger.error(f"Fail to clip for grid {grid_id}: {e}", exc_info=True)
        continue


# with open("not_fully_contained_grids_one.txt", 'w') as f:
#    for grid in not_fully_contained_one:
#        f.write(str(grid) + "\n")
#
#
# with open("not_fully_contained_grids_several.txt", 'w') as f:
#    for grid in not_fully_contained_several:
#        f.write(str(grid) + "\n")

    # bounds = grid_polygon.bounds
    # transform = from_bounds(*bounds, width=IMAGE_SIZE, height=IMAGE_SIZE)
    
    # samples_with_value = [
    #     (row.geometry, row.TreeAI_ID) 
    #     for _, row in samples_within_grid.iterrows()
    # ]
    
    # output_path = out_masks_dir / f"{int(grid_id):04d}.tif"
    
    # with rasterio.open(
    #     output_path,
    #     'w',
    #     driver='GTiff',
    #     height=IMAGE_SIZE,
    #     width=IMAGE_SIZE,
    #     count=1,
    #     dtype=rasterio.uint8,
    #     crs=fishnet_df.crs,
    #     transform=transform,
    #     nodata=255
    # ) as dst:
    #     burned_array = features.rasterize(
    #         shapes=samples_with_value,
    #         out_shape=(IMAGE_SIZE, IMAGE_SIZE),
    #         transform=transform,
    #         fill=0,
    #         all_touched=True,
    #         dtype=rasterio.uint8
    #     )

    #     dst.write(burned_array, 1)