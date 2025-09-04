import os
import geopandas as gpd
import rasterio
from pathlib import Path
from shapely.geometry import box, Point
import random
from rasterio import CRS
from rasterio.features import rasterize
from rasterio.windows import Window
from tqdm import tqdm
import logging
import concurrent.futures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(processName)s - %(message)s',
    filename='runs/create_patch_log.txt',
    filemode='w'
)


PATCH_SIZE_PIX = 500
MAX_ITER = 1000
MAX_WORKERS = 2
SWISS_RASTER_ROOT = Path('X:/Dead/raster/01_switzerland')
PATCH_SAVE_DIR = Path("data")
ANNO_SHP_PATH = 'data/anno_shp/Data_Swiss_NDVI_XY_50buf_up_28782_LF_MB_ZX_27790spDead.shp'


def create_search_box(center_point: Point, pixel_size, patch_size_pix, rand_offset_ratio=0.2):
    half_size = patch_size_pix * pixel_size / 2
    offset_x = random.uniform(-rand_offset_ratio * half_size, rand_offset_ratio * half_size)
    offset_y = random.uniform(-rand_offset_ratio * half_size, rand_offset_ratio * half_size)
    min_x, max_x = center_point.x - half_size + offset_x, center_point.x + half_size + offset_x
    min_y, max_y = center_point.y - half_size + offset_y, center_point.y + half_size + offset_y
    search_box = box(min_x, min_y, max_x, max_y)
    return search_box


def clip_raster(raster, clip_range: box, box_crs: CRS, output, patch_size_pix=PATCH_SIZE_PIX, nodata=255):
    assert raster.crs == box_crs, "The CRS of input raster should be the same as the CRS of clip raster"
    center_point = clip_range.centroid
    center_row, center_col = raster.index(center_point.x, center_point.y)
    half_pix = patch_size_pix // 2

    patch_window = Window(
        col_off=center_col - half_pix,
        row_off=center_row - half_pix,
        width=patch_size_pix,
        height=patch_size_pix
    )
    out_image = raster.read(
        window=patch_window,
        boundless=True,
        fill_value=nodata
    )
    out_transform = raster.window_transform(patch_window)

    out_meta = raster.meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform
    })

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, "w", **out_meta) as dest:
        dest.write(out_image)

    return out_meta


def rasterize_shp_label(labels_gdf: gpd.GeoDataFrame, out_meta: dict, output, background=0):
    shapes = [(geom, tree_id) for geom, tree_id in zip(labels_gdf.geometry, labels_gdf.TreeAI_ID)]
    mask_image = rasterize(
        shapes=shapes,
        out_shape=(out_meta['height'], out_meta['width']),
        transform=out_meta['transform'],
        fill=background,
        dtype=rasterio.uint8
    )

    mask_meta = out_meta.copy()
    mask_meta.update(count=1, dtype='uint8')

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output, 'w', **mask_meta) as dst:
        dst.write(mask_image, 1)


def create_patch_by_tif(tiff_name, anno_group, swiss_raster_root=SWISS_RASTER_ROOT,
                        patch_save_dir=PATCH_SAVE_DIR, max_iter=MAX_ITER):
    try:
        anno_tiff = anno_group.copy()
        anno_tiff['covered'] = 0

        base_raster_path = swiss_raster_root / tiff_name
        if not base_raster_path.exists():
            logging.warning(f"Tiff file not found: {base_raster_path}")
            return

        with rasterio.open(base_raster_path) as base_img:
            pixel_size_x, pixel_size_y = base_img.res

            if pixel_size_x != pixel_size_y:
                logging.error(f"Failed to process file '{tiff_name}': X and Y resolutions do not match "
                f"({pixel_size_x} != {pixel_size_y}). Skipping this file.")
                return

            iter = 0
            while len(anno_tiff[(anno_tiff['covered'] == 0)]):
                polyg = anno_tiff[
                    (anno_tiff['covered'] == 0)].sample()  # Randomly select one polygon annotations as the center
                center_pt = polyg.iloc[0]['centroid']
                search_box = create_search_box(center_point=center_pt,
                                            pixel_size=pixel_size_x,
                                            patch_size_pix=PATCH_SIZE_PIX)

                condition = anno_tiff.within(search_box)
                samples = anno_tiff[condition]
                # indices_to_update = anno_within[condition].index
                if not samples.empty:
                    anno_tiff.loc[condition, 'covered'] = 1

                    patch_img_output = patch_save_dir / (f'images/{Path(tiff_name).stem}/' + f'{iter}.tif')
                    patch_label_output = patch_save_dir / (f'masks/{Path(tiff_name).stem}/' + f'{iter}.tif')

                    out_mata = clip_raster(base_img, search_box, base_img.crs, output=patch_img_output)
                    rasterize_shp_label(samples, out_meta=out_mata, output=patch_label_output)
                else:
                    logging.warning(f"For {tiff_name}, a search box was created with no polygons inside.")

                if iter < max_iter:
                    iter += 1
                else:
                    logging.info(f"For {tiff_name}, iteration has reached maximum number-({max_iter}) !")
                    break
    
    except Exception as e:
        logging.error(f"Prolem arose when processing '{tiff_name}': {e}")


def process_tiff_wrapper(args):
    tiff_name, anno_group = args
    create_patch_by_tif(tiff_name, anno_group)


if __name__ == '__main__':
    core_count = os.cpu_count()
    logging.info(f"This computer has {core_count} CPU cores.")

    anno_df = gpd.read_file(ANNO_SHP_PATH)
    anno_df = anno_df.explode(ignore_index=True)  # Explode MUTIPOLYGON to POLYGON
    anno_df['centroid'] = anno_df.geometry.centroid
    grouped_annos = anno_df.groupby('Tiff')
    tiff_groups = [(name, group) for name, group in grouped_annos]
    tiff_groups_desc = sorted(tiff_groups, key=lambda item: len(item[1]), reverse=True)

    logging.info(f"Ready to process {len(tiff_groups)} Tiff files...")

    for tiff_group in tqdm(tiff_groups_desc, total=len(tiff_groups_desc), desc='Processing Tiff files'):
        process_tiff_wrapper(tiff_group)
    
    # with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    #     list(tqdm(executor.map(process_tiff_wrapper, tiff_groups_desc), total=len(tiff_groups_desc))
    
    logging.info("Jobs Done!")
