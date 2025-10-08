from rasterio.transform import from_bounds
from pathlib import Path
from tqdm import tqdm
import rasterio
from rasterio.windows import from_bounds, transform as win_transform
from rasterio import CRS


images_folder = Path("../data/TreeAI_Swiss/images")
images = list(images_folder.rglob("*.tif"))


def clip_raster_with_template(in_raster_path: str, template_path: str, out_raster_path: str):
    with rasterio.open(in_raster_path) as src_sat, rasterio.open(template_path) as tpl:
        src_crs = src_sat.crs
        tpl_crs = tpl.crs
        if src_sat.crs != tpl.crs:
            raise ValueError(f"CRS mismatch: {src_sat.crs} vs {tpl.crs}. Reproject first or use method B.")

        # window of template bounds in source grid
        win = from_bounds(*tpl.bounds, transform=src_sat.transform)
        win = win.round_offsets().round_lengths()  # align to pixel grid

        data = src_sat.read(window=win, boundless=True, fill_value=src_sat.nodata)
        transform = win_transform(win, src_sat.transform)

        assert data.shape[1] == 5 and data.shape[2] == 5, "Size of clipped raster should be 5x5"

        profile = src_sat.profile.copy()
        profile.update(height=data.shape[1],
                       width=data.shape[2],
                       transform=transform,
                       crs=CRS.from_wkt(tpl.crs.to_wkt()))

        # with rasterio.open(out_raster_path, "w", **profile) as dst:
        #     dst.write(data)


for image in tqdm(images, desc="Clipping Rasters"):
    sate_embed_path = str(image).replace("images", "Satellite_Embedding")

    template_path = str(image)

    output_path = Path(sate_embed_path.replace("Satellite_Embedding", "clip_temp"))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    clip_raster_with_template(
        in_raster_path=sate_embed_path,
        template_path=template_path,
        out_raster_path=str(output_path)
    )
