from pathlib import Path
import os
import shutil
import calendar

target_folder = "data/TreeAI_Swiss/DSM"
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


# ----------------------------------------------------------
# Below are script for arranging the raw Sentinel-1 folder
# ----------------------------------------------------------

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