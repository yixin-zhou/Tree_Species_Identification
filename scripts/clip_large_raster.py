# Based on ArcGIS Pro
import arcpy
from pathlib import Path
from tqdm import tqdm
import shutil

arcpy.env.pyramid = "NONE"
arcpy.env.overwriteOutput = True
arcpy.env.rasterStatistics = "NONE"


images_folder = Path("../data/TreeAI_Swiss/images")
images = list(images_folder.rglob("*.tif"))

for image in tqdm(images, desc="Clipping Rasters"):
    sate_embed_path = str(image).replace("images", "Satellite_Embedding")
    template_path = str(image)
    output_path = Path(sate_embed_path.replace("Satellite_Embedding", "clip_temp"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    arcpy.management.Clip(
        in_raster=sate_embed_path,
        rectangle="#",
        out_raster=str(output_path),
        in_template_dataset=template_path,
        maintain_clipping_extent="MAINTAIN_EXTENT"
    )
