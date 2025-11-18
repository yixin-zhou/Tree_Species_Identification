from pathlib import Path
import numpy as np
import rasterio
from rasterio.coords import BoundingBox
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from tqdm import tqdm
import calendar
import re


# def find_containing_patch(tif_path: BoundingBox, patch_bounds: dict):
#     with rasterio.open(tif_path) as src:
#         tif_bounds = src.bounds
#
#     for patch_id, patch_bbox in patch_bounds.items():
#         is_contained = (
#                 patch_bbox.left <= tif_bounds.left and
#                 patch_bbox.right >= tif_bounds.right and
#                 patch_bbox.bottom <= tif_bounds.bottom and
#                 patch_bbox.top >= tif_bounds.top
#         )
#         if is_contained:
#             return patch_id
#
#     return None
#
#
# def reproject_satellite(
#     sat_path: str, uav_path: str, out_path: str,
#     resampling: Resampling = Resampling.bilinear
# ):
#     with rasterio.open(sat_path) as src, rasterio.open(uav_path) as uav:
#         xmin, ymin, xmax, ymax = uav.bounds
#
#         dst_width, dst_height = 5, 5
#         dst_transform = from_bounds(xmin, ymin, xmax, ymax, dst_width, dst_height)
#         dst_crs = src.crs
#
#         dst = np.full((src.count, dst_height, dst_width),
#                       src.nodata if src.nodata is not None else 0,
#                       dtype=src.dtypes[0])
#
#         for b in range(1, src.count + 1):
#             reproject(
#                 source=rasterio.band(src, b),
#                 destination=dst[b-1],
#                 src_transform=src.transform,
#                 src_crs=src.crs,
#                 dst_transform=dst_transform,
#                 dst_crs=dst_crs,
#                 dst_nodata=src.nodata,
#                 resampling=resampling
#             )
#
#         profile = src.profile.copy()
#         profile.update({
#             "height": dst_height,
#             "width":  dst_width,
#             "transform": dst_transform,
#             "crs": dst_crs,
#             "nodata": src.nodata,
#             "compress": profile.get("compress", "DEFLATE"),
#         })
#
#         with rasterio.open(out_path, "w", **profile) as dst_ds:
#             dst_ds.write(dst)
#
#
# def clip_Sentinel(sentinel_path, img_path, savepath, pad_pixel=1):
#     with rasterio.open(sentinel_path) as sentinel, rasterio.open(img_path) as img:
#         clip_bounds = img.bounds
#
#
#
# def check_bounds(setinel_path, image_path):
#     with rasterio.open(setinel_path) as sentinel, rasterio.open(image_path) as image:
#         assert sentinel.crs == image.crs, "The CRS of Sentinel image and UAV image should be the same"
#         sentinel_bound = sentinel.bounds
#         image_bound = image.bounds
#
#     contains = (
#         (sentinel_bound.left <= image_bound.left) and
#         (sentinel_bound.bottom <= image_bound.bottom) and
#         (sentinel_bound.right >= image_bound.right) and
#         (sentinel_bound.top >= image_bound.top)
#     )
#
#     return contains
#
#
#
# if __name__ == '__main__':
#     images_folder = Path("../data/TreeAI_Swiss/images")
#     patch_bounds = parse_bounds_file("../data/raw/Sentinel-1/GEE_patch_bound.txt")
#     for image in tqdm(list(images_folder.rglob("*.tif")), desc="Reproject to UAV extent"):
#         # sat_path = str(image).replace("images", "Satellite_Embedding")
#         # uav_path = str(image)
#         # out_path = Path(sat_path.replace("Satellite_Embedding", "clip_temp"))
#         # out_path.parent.mkdir(parents=True, exist_ok=True)
#         #
#         # reproject_satellite(
#         #     sat_path, uav_path, str(out_path),
#         #     resampling=Resampling.nearest
#         # )
#
#         capture_year = int(image.name[:4])
#         for month in range(1, 13):
#             sentinel_folder = Path(f"../data/raw/Sentinel-1/{capture_year}_{calendar.month_name[month]}")
#             sentinel_patches = list(sentinel_folder.rglob("*.tif"))
#
#             patch_number = find_containing_patch(image, patch_bounds)
#             sentinel_patch = [f for f in sentinel_patches if patch_number in f.name]
#             assert len(sentinel_patch) == 1, "There should be only one matched sentinel file"



numbers = ['0000000000-0000000000', '0000000000-0000016384', '0000000000-0000032768',
          '0000016384-0000000000', '0000016384-0000016384', '0000016384-0000032768']
bounds = {key: [] for key in numbers}

for year in range(2017, 2023):
    for month in range(1, 13):
        sentinel_folder = Path(f"../data/raw/Sentinel-1/{year}_{calendar.month_name[month]}")
        sentinel_patches = list(sentinel_folder.rglob("*.tif"))
        for number in numbers:
            file = [f for f in sentinel_patches if number in f.name]
            assert len(file) == 1, "no match number"
            with rasterio.open(file[0]) as src:
                bounds[number].append(src.bounds)

with open("GEE_patch_bound.txt", 'w') as f:
    for number in numbers:
        assert len(bounds[number]) == 72, f"Len should be 72, instead of {len(bounds[number])}"
        bound_set = set(bounds[number])
        assert len(bound_set) == 1, f"There are several range for {number}"
        f.write(f"{number}: {bound_set}\n")
