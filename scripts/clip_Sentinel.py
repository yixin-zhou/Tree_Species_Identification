from pathlib import Path
import geopandas as gpd
import rasterio
from rasterio.coords import BoundingBox
from Utils.utils import getAcquireYear
from tqdm import tqdm
import calendar
import re
import subprocess
import shlex


def parse_bounds_file(file_path: Path) -> dict:
    bounds_dict = {}
    pattern = re.compile(
        r"BoundingBox\(left=([\d.]+), bottom=([\d.]+), right=([\d.]+), top=([\d.]+)\)"
    )

    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or ':' not in line:
                continue

            parts = line.split(':', 1)
            patch_id = parts[0].strip()
            bound_str = parts[1].strip()

            match = pattern.search(bound_str)
            if match:
                left, bottom, right, top = map(float, match.groups())
                bounds_dict[patch_id] = BoundingBox(left=left, bottom=bottom, right=right, top=top)

    return bounds_dict


def find_containing_patch(tif_path: BoundingBox, patch_bounds: dict):
    with rasterio.open(tif_path) as src:
        tif_bounds = src.bounds

    for patch_id, patch_bbox in patch_bounds.items():
        is_contained = (
                patch_bbox.left <= tif_bounds.left and
                patch_bbox.right >= tif_bounds.right and
                patch_bbox.bottom <= tif_bounds.bottom and
                patch_bbox.top >= tif_bounds.top
        )
        if is_contained:
            return patch_id

    return None

ANNOTATIONS_SHP = "../data/shapefile/grid/swiss_tree_annotations_with_filtered_grid.shp"
images_folder = Path("../data/TreeAI_Swiss_60/masks")
patch_bounds = parse_bounds_file("GEE_patch_bound.txt")

anno_df = gpd.read_file(ANNOTATIONS_SHP)

for image in tqdm(list(images_folder.rglob("*.tif")), desc="Reproject to UAV extent"):
    # if image.name != "20180620_1332_12501_0_13_17.tif":
    #     continue
    capture_year = getAcquireYear(int(image.stem), anno_df)
    for month in range(1, 13):
        try:
            sentinel_folder = Path(f"../data/raw/Sentinel-1/{capture_year}_{calendar.month_name[month]}")
            sentinel_patches = list(sentinel_folder.rglob("*.tif"))
            assert len(sentinel_patches) == 6, "There should be six sentinel patches"

            save_dir = Path(str(image).replace('masks', 'Sentinel-1').replace('.tif', ''))
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / f"{calendar.month_name[month]}.tif"
            if save_path.exists():
                continue

            patch_number = find_containing_patch(image, patch_bounds)
            sentinel_patch = [f for f in sentinel_patches if patch_number in f.name]
            assert len(sentinel_patch) == 1, "There should be only one matched sentinel file"

            with rasterio.open(image) as src:
                target_bounds = src.bounds
                xmin, ymin, xmax, ymax = target_bounds
                image_crs = src.crs.to_string()

            command = (
                f"gdalwarp -overwrite -te {xmin} {ymin} {xmax} {ymax} "
                f"-ts 6 6 "
                f"-r bilinear "
                f"-t_srs EPSG:2056 "
                f"\"{str(sentinel_patch[0])}\" "
                f"\"{str(save_path)}\""
            )

            process = subprocess.run(shlex.split(command), capture_output=True, text=True)

            if process.returncode != 0:
                print(f"Error fo {image.name}: {process.stderr}")

        except Exception as e:
            print(f"Error fo {image.name}: {e}")