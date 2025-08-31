<<<<<<< HEAD
import geopandas as gpd
import rasterio
from pathlib import Path
from shapely.geometry import box, Point
import random
from rasterio import CRS
from rasterio.features import rasterize
from rasterio.windows import Window
from tqdm import tqdm


PATCH_SIZE_PIX = 600


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

if __name__ == '__main__':

    PATCH_SIZE_PIX = 600

    swiss_raster_root = Path('data')
    anno_shp_path = 'data/TreeAI_Swiss/annotations_shapefile/Data_Swiss_NDVI_XY_50buf_up_28782_LF_MB_ZX_27790spDead.shp'
    anno_df = gpd.read_file(anno_shp_path)

    anno_df = anno_df.explode(ignore_index=True)  # Explode MUTIPOLYGON to POLYGON
    anno_df['centroid'] = anno_df.geometry.centroid

    anno_crs = anno_df.crs

    tiff_counts = anno_df['Tiff'].value_counts()
    anno_df['covered'] = 0
    tiff_num = len(tiff_counts)

    patches_save_dir = Path("data/test")
    for tiff_name, counts in tqdm(tiff_counts.items(), desc="Clipping images to patches", total=tiff_num):
        if tiff_name != '20220717_0737_12501_30_0.tif':
            continue

        anno_within = anno_df[anno_df['Tiff'] == tiff_name].copy()
        base_raster_path = swiss_raster_root / tiff_name
        with rasterio.open(base_raster_path) as base_img:
            pixel_size_x, pixel_size_y = base_img.res
            assert pixel_size_x == pixel_size_y, "The resolution of X and Y should be the same."

            samples_num = []
            iter = 0
            while len(anno_within[(anno_within['covered'] == 0)]):
                polyg = anno_within[
                    (anno_within['covered'] == 0)].sample()  # Randomly select one polygon annotations as the center
                center_pt = polyg.iloc[0]['centroid']
                search_box = create_search_box(center_point=center_pt,
                                               pixel_size=pixel_size_x,
                                               patch_size_pix=PATCH_SIZE_PIX)
                condition = anno_within.within(search_box)
                samples = anno_within[condition]
                indices_to_update = anno_within[condition].index

                if len(indices_to_update) > 0:
                    anno_within.loc[condition, 'covered'] = 1
                    anno_df.loc[indices_to_update, 'covered'] = 1

                    patch_img_output = patches_save_dir / ('images/' + Path(tiff_name).stem + f'_{iter}.tif')
                    patch_label_output = patches_save_dir / ('masks/' + Path(tiff_name).stem + f'_{iter}.tif')

                    out_mata = clip_raster(base_img, search_box, base_img.crs, output=patch_img_output)
                    rasterize_shp_label(samples, out_meta=out_mata, output=patch_label_output)
                else:
                    print(f"There is no polygons within {tiff_name}")

                iter += 1

=======
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
>>>>>>> e7e640b071f84c7a093ae1d39e9f5158e653688d
