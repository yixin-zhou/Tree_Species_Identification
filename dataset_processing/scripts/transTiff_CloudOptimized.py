from rio_cogeo.cogeo import cog_translate
from rio_cogeo.profiles import cog_profiles
from pathlib import Path
from tqdm import tqdm
import os

# The main purpose of this script is to translate raster files in .tif format to cloud optimized ones

def to_cog(src_tif, dst_tif):
    profile = cog_profiles.get("deflate")
    profile.update(dict(BLOCKSIZE=512, BIGTIFF="IF_SAFER"))

    cog_translate(
        src_tif,
        dst_tif,
        profile,
        in_memory=False,
        overview_level=5,
        overview_resampling="average",
        web_optimized=False,
        forward_band_tags=True,
        quiet=True,
    )

if __name__ == '__main__':
    search_dir = Path("../data/TreeAI_Swiss_60/masks/")
    remove_origin = True
    
    uav_images = list(search_dir.rglob("*.tif"))
    
    for file_path in tqdm(uav_images, desc="Transferring .tif format raster files to cloud optimized ones"):
        cog_filename = f"{file_path.stem}_COG{file_path.suffix}"
        dst_filepath = file_path.with_name(cog_filename)
        to_cog(src_tif=str(file_path), dst_tif=dst_filepath)
        if remove_origin:
            os.remove(str(file_path))
            os.rename(str(dst_filepath), str(file_path))
