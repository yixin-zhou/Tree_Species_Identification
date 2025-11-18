import requests
import numpy as np
import warnings
import random
import os
import torch
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.enums import ColorInterp
from pathlib import Path


warnings.filterwarnings("ignore", category=UserWarning, module="rasterio.windows")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="rasterio.windows")


# https://github.com/qhd1996/seed-everything
def seed(TORCH_SEED):
    random.seed(TORCH_SEED)
    os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
    np.random.seed(TORCH_SEED)
    torch.manual_seed(TORCH_SEED)
    torch.cuda.manual_seed_all(TORCH_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def determine_within(in_bounds, out_bounds):
    out_left, out_bottom, out_right, out_top = out_bounds
    in_left, in_bottom, in_right, in_top = in_bounds
    
    return (
        (in_left   >= out_left) and
        (in_right  <= out_right) and
        (in_bottom >= out_bottom) and
        (in_top    <= out_top)
    )
    

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


def clip(input_raster, clip_raster, output, nodata=65535):
    with rasterio.open(clip_raster) as c:
        clip_crs = c.crs
        clip_range = c.bounds

    with rasterio.open(input_raster) as src:
        image_dtype = src.dtypes[0]
        assert image_dtype == 'uint16', "The dtype of clippped image should be uint16."
        input_crs = src.crs
        input_raster_range = src.bounds
            
        assert input_crs == clip_crs, "The CRS of the input raster and clipped raster should be the same."
        clip_window_pre = from_bounds(*clip_range, src.transform)

        start_col = clip_window_pre.col_off
        start_row = clip_window_pre.row_off

        clip_window = Window(
            col_off=start_col, 
            row_off=start_row, 
            width=600, 
            height=600
        )
        
        clipped_data = src.read(
            window=clip_window, 
            boundless=True,
            fill_value=nodata
        )
        
        out_meta = src.meta.copy()

        out_transform = src.window_transform(clip_window)

        out_meta.update({
            "driver": "GTiff",
            "height": clipped_data.shape[1],
            "width": clipped_data.shape[2],
            "transform": out_transform,
            "nodata":nodata
        })

        with rasterio.open(output, "w", **out_meta) as dest:
            dest.write(clipped_data)
        
        return determine_within(in_bounds=clip_range, out_bounds=input_raster_range)


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


def mosaic_clip(input_rasters, clip_raster, output, nodata=65535, target_res=0.1):
    with rasterio.open(clip_raster) as c:
        clip_crs = c.crs
        clip_bounds = c.bounds
        left, bottom, right, top = clip_bounds
        width = height = 600
        dst_transform = transform_from_bounds(left, bottom, right, top, width, height)

    with rasterio.open(input_rasters[0]) as src0:
        profile = src0.profile.copy()
        profile.update(
            height=height,
            width=width,
            transform=dst_transform,
            crs=clip_crs,
        )

    # fill_val = nodata if nodata is not None else 0
    mosaic = np.full((4, height, width), nodata, dtype=np.dtype("uint16"))

    for p in input_rasters:
        with rasterio.open(p) as src:
            image_dtype = src.dtypes[0]
            assert image_dtype == 'uint16', "The dtype of clippped image should be uint16."
            assert src.crs == profile["crs"], "CRS is different"
            
            if not (abs(src.res[0] - target_res) < 1e-5 and abs(src.res[1] - target_res) < 1e-5):
                print(f"Resolution of {p} is X={src.res[0]}, Y={src.res[1]}")
            
            with WarpedVRT(
                src,
                crs=profile["crs"],
                transform=dst_transform,
                width=width,
                height=height,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=nodata,
            ) as vrt:
                arr = vrt.read()
                src_mask = ~(np.all(arr == nodata, axis=0))
                dst_mask = np.all(mosaic == nodata, axis=0)
                write_mask = dst_mask & src_mask
                for i in range(4):
                    mosaic[i][write_mask] = arr[i][write_mask]

    # bottom, left, top, right = array_bounds(profile["height"], profile["width"], profile["transform"])
    # mosaic_bounds = (left, bottom, right, top)

    with rasterio.open(output, "w", **profile) as dst:
        dst.write(mosaic)

    return True


def add_overvier2GeoTiff(filepath, levels=[2, 4]):
    with rasterio.open(filepath, 'r+') as dst:
        dst.build_overviews(levels, Resampling.average)
        dst.update_tags(ns='rio_overview', build=True)

        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "Green")
        dst.set_band_description(3, "Blue")

        dst.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.undefined,
        )

        dst.close()
        return True


def getAcquireYear(grid_id, anno_df):
    samples = anno_df[(anno_df['grid_id'] == grid_id)]
    years = samples['Year'].unique()
    if len(years) == 1:
        return years[0]
    else:
        # print(f"There are several acquiring years for this grid {grid_id}, including {years}")
        return np.max(years)


def get_divice() -> torch.device:
    '''
    Automatically determine on which device should the training script implements
    '''
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

