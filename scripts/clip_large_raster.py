from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
from tqdm import tqdm


def reproject_satellite(
    sat_path: str, uav_path: str, out_path: str,
    resampling: Resampling = Resampling.bilinear
):
    with rasterio.open(sat_path) as src, rasterio.open(uav_path) as uav:
        xmin, ymin, xmax, ymax = uav.bounds

        dst_width, dst_height = 5, 5
        dst_transform = from_bounds(xmin, ymin, xmax, ymax, dst_width, dst_height)
        dst_crs = src.crs

        dst = np.full((src.count, dst_height, dst_width),
                      src.nodata if src.nodata is not None else 0,
                      dtype=src.dtypes[0])

        for b in range(1, src.count + 1):
            reproject(
                source=rasterio.band(src, b),
                destination=dst[b-1],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=dst_crs,
                dst_nodata=src.nodata,
                resampling=resampling
            )

        profile = src.profile.copy()
        profile.update({
            "height": dst_height,
            "width":  dst_width,
            "transform": dst_transform,
            "crs": dst_crs,
            "nodata": src.nodata,
            "compress": profile.get("compress", "DEFLATE"),
        })

        with rasterio.open(out_path, "w", **profile) as dst_ds:
            dst_ds.write(dst)


if __name__ == '__main__':
    images_folder = Path("../data/TreeAI_Swiss/images")
    for image in tqdm(list(images_folder.rglob("*.tif")), desc="Reproject to UAV extent"):
        sat_path = str(image).replace("images", "Satellite_Embedding")
        uav_path = str(image)
        out_path = Path(sat_path.replace("Satellite_Embedding", "clip_temp"))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        reproject_satellite(
            sat_path, uav_path, str(out_path),
            resampling=Resampling.nearest
        )