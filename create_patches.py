import geopandas as gpd
import rasterio
from pathlib import Path
from shapely.geometry import box, Point
import random


def create_search_box(center_point: Point, pixel_size, patch_size_pix, rand_offset_ratio=0.2):
    half_size = patch_size_pix * pixel_size / 2
    offset_x = random.uniform(-rand_offset_ratio * half_size, rand_offset_ratio * half_size)
    offset_y = random.uniform(-rand_offset_ratio * half_size, rand_offset_ratio * half_size)
    min_x, max_x = center_point.x - half_size + offset_x, center_point.x + half_size + offset_x
    min_y, max_y = center_point.y - half_size + offset_y, center_point.y + half_size + offset_y
    search_box = box(min_x, min_y, max_x, max_y)
    return search_box


patch_size_pix = 600

swiss_raster_root = Path('data')
anno_shp_path = 'data/TreeAI_Swiss/annotations_shapefile/Data_Swiss_NDVI_XY_50buf_up_28782_LF_MB_ZX_27790spDead.shp'
anno_df = gpd.read_file(anno_shp_path)

anno_df = anno_df.explode(ignore_index=True)  # Explode MUTIPOLYGON to POLYGON
anno_df['centroid'] = anno_df.geometry.centroid

anno_crs = anno_df.crs

tiff_counts = anno_df['Tiff'].value_counts()
anno_df['covered'] = 0

for tiff_name, counts in tiff_counts.items():
    if tiff_name != '20220717_0737_12501_30_0.tif':
        continue

    anno_within = anno_df[anno_df['Tiff'] == tiff_name].copy()
    base_raster_path = swiss_raster_root / tiff_name
    with rasterio.open(base_raster_path) as base_img:
        pixel_size_x, pixel_size_y = base_img.res
        assert pixel_size_x == pixel_size_y, "The resolution of X and Y should be the same."

    samples_num = []
    iter = 1
    while len(anno_within[(anno_within['covered'] == 0)]):
        polyg = anno_within[
            (anno_within['covered'] == 0)].sample()  # Randomly select one polygon annotations as the center
        center_pt = polyg.iloc[0]['centroid']
        search_box = create_search_box(center_point=center_pt,
                                       pixel_size=pixel_size_x,
                                       patch_size_pix=patch_size_pix)
        condition = anno_within.within(search_box)
        samples = anno_within[condition]
        anno_within.loc[condition, 'covered'] = 1

        left_samples_num = len(anno_within[(anno_within['covered'] == 0)])
        samples_num.append(len(samples))
        print(f'Iteration {iter}, update {len(samples)} samples, {left_samples_num} samples left')
        iter += 1