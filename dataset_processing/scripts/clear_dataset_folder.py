from pathlib import Path
import rasterio
import os
import shutil
import calendar
from tqdm import tqdm


# --------------------------------------------------------------------
# Below are script for deleting non-tif files in the target folder
# --------------------------------------------------------------------
# target_folder = "../data/TreeAI_Swiss_60/masks"
# target_folder = Path(target_folder)
#
# tif_files = list(target_folder.rglob("*.tif"))
# for tif_file in tqdm(tif_files):
#     with rasterio.open(tif_file) as src:
#         w, h = src.width, src.height
#         transform = src.transform
#         pixel_width = transform.a
#         pixel_height = transform.e
#     if w != 600 or h != 600:
#         print(f"{tif_file.name} ({w}x{h}), pixel size is ({pixel_width}, {pixel_height})")
#         # tif_file.unlink(missing_ok=True)


# --------------------------------------------------------------------
# Below are script for deleting non-tif files in the target folder
# --------------------------------------------------------------------

target_folder = "../data/TreeAI_Swiss_60/VHM"
target_folder = Path(target_folder)

non_tif_files = [f for f in target_folder.rglob("*") if f.is_file() and f.suffix.lower() != ".tif"]
for f in non_tif_files:
    print(f)

flag = input("Do you want to delete all the non-tif files? (0-No, 1-Yes): ")
if flag.strip() == "1":

    for f in non_tif_files:
        try:
            f.unlink()
            print(f"Delete: {f}")
        except Exception as e:
            print(f"Fail to delete {f}: {e}")
else:
    print("Did not delete any non-tif file.")



# --------------------------------------------------------------------
# Below are script for arranging the raw Sentinel-1 folder
# --------------------------------------------------------------------

# root_dir = Path("../data/raw/Sentinel-1")
# sentinel_images = list(root_dir.rglob("*.tif"))
# for year in range(2017, 2023):
#     for month in range(1, 13):
#         new_folder = Path(f"../data/raw/Sentinel-1/{year}_{calendar.month_name[month]}")
#         new_folder.mkdir(parents=True, exist_ok=True)
#
#         sentinel_within_month = [f for f in sentinel_images if str(year) in f.name and calendar.month_name[month] in f.name]
#         print(f"In {calendar.month_name[month]}, {year}, there are {len(sentinel_within_month)} sentinel-1 images")
#         for f in sentinel_within_month:
#             filename = f.name
#             dst_path = new_folder / filename
#             shutil.move(src=f, dst=dst_path)


# --------------------------------------------------------------------
# Below are script for arranging the raw Sentinel-1 folder
# --------------------------------------------------------------------
# modalities = ['images', 'DEM', 'DSM', 'VHM', 'Sentinel-1', 'Sentinel-2',
#             'Satellite_Embedding', 'climate', 'labels', 'masks', 'images_png']
#
# TreeAI_DIR = '../data/TreeAI_Swiss_60/'
# with open('../data/problem_grid.txt', 'r') as f:
#     grids = f.readlines()

# for grid in tqdm(grids, desc='Removing abnormal grids'):
#     for modal in modalities:
#         if modal == 'images_png':
#             suffix = '.png'
#         elif modal in ['labels', 'climate']:
#             suffix = '.txt'
#         else:
#             suffix = '.tif'
#         grid_id = grid.replace('\n', '') + suffix
#         delete_file = os.path.join(TreeAI_DIR, modal, grid_id)
#         if os.path.exists(delete_file):
#             os.remove(delete_file)

# for grid in tqdm(grids, desc='Removing abnormal grids for Sentinel-1'):
#     grid_id = grid.replace('\n', '')
#     delete_folder = os.path.join(TreeAI_DIR, 'Sentinel-1', grid_id)
#     shutil.rmtree(delete_folder)
