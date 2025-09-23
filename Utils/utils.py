import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
from pathlib import Path
import requests
from tqdm import tqdm
import os

def download_file(url: str, save_path: str, chunk_size: int = 1024):
    save_path = Path(save_path)

    if save_path.is_dir():
        filename = url.split("/")[-1]
        save_path = save_path / filename

    response = requests.get(url, stream=True)
    response.raise_for_status()

    with open(save_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)


def clip(input_raster, clip_raster, output):
    with rasterio.open(clip_raster) as c:
        clip_crs = c.crs
        clip_range = c.bounds

    with rasterio.open(input_raster) as src:
        input_crs = src.crs
        assert input_crs == clip_crs, "The CRS of the input raster and clipped raster should be the same."
        clip_window = from_bounds(*clip_range, src.transform)

        clipped_data = src.read(window=clip_window)
        out_meta = src.meta.copy()

        out_transform = src.window_transform(clip_window)

        out_meta.update({
            "driver": "GTiff",
            "height": clipped_data.shape[1],
            "width": clipped_data.shape[2],
            "transform": out_transform
        })

        with rasterio.open(output, "w", **out_meta) as dest:
            dest.write(clipped_data)


def mosaic_clip(input_rasters, clip_raster, output):
    temp_mosaic_path = os.path.join(os.path.dirname(output), 'temp.tif')

    with rasterio.open(clip_raster) as c:
        clip_crs = c.crs

    mosaic_rasters = []
    for input_raster in input_rasters:
        raster = rasterio.open(str(input_raster))
        assert raster.crs == clip_crs, "The CRS of the input rasters and clipped raster should be the same."
        mosaic_rasters.append(raster)

    mosaic, out_trans = merge(mosaic_rasters)

    out_meta = mosaic_rasters[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_trans,
    })

    with rasterio.open(temp_mosaic_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    for src in mosaic_rasters:
        src.close()

    clip(input_raster=temp_mosaic_path, clip_raster=clip_raster, output=output)
    delete_files([pathlib.Path(temp_mosaic_path)])


def delete_files(files):
    for file in files:
        if file.exists() and file.is_file():
            file.unlink()


def get_raster_size(raster_filepath):
    with rasterio.open(raster_filepath) as src:
        pixel_size_x, pixel_size_y = src.res
        return {
            "width": src.width,
            "height": src.height,
            "pixel_size_x": pixel_size_x,
            "pixel_size_y": pixel_size_y
        }