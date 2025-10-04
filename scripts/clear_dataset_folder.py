from pathlib import Path
import os

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

