<<<<<<< HEAD
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
import pathlib
from tqdm import tqdm
import os


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


if __name__ == '__main__':
    treeai_swiss_folder = pathlib.Path('data/Swiss/TreeAI_Swiss')
    folders = treeai_swiss_folder.iterdir()

    for folder in tqdm(folders, desc=f"Clipping height models"):
        dem_rasters = list(folder.glob("*swissalti3d*.tif"))
        dsm_rasters = list(folder.glob("*swisssurface3d*.tif"))

        assert len(dem_rasters) == len(dsm_rasters), "The length of DEM and DSM products should be the same."

        uav_raster = str(folder / (folder.name + '.tif'))
        raster_size_compare= []

        for height_model in ['dem', 'dsm']:
            clipped_height_model = str(folder / (folder.name + f'_{height_model}.tif'))
            height_model_rasters = dem_rasters if height_model == 'dem' else dsm_rasters

            if len(height_model_rasters) == 1:
                clip(input_raster=str(height_model_rasters[0]), clip_raster=uav_raster, output=clipped_height_model)
            else:
                mosaic_clip(input_rasters=height_model_rasters, clip_raster=uav_raster, output=clipped_height_model)

            raster_size = get_raster_size(clipped_height_model)
            raster_size_compare.append(raster_size)

            delete_files(height_model_rasters)
        if raster_size_compare[0] != raster_size_compare[1]:
            raise ValueError(f"Raster size of {folder.name} is different")



=======
import rasterio
from rasterio.windows import from_bounds
from rasterio.merge import merge
import pathlib
from tqdm import tqdm
import os


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


if __name__ == '__main__':
    treeai_swiss_folder = pathlib.Path('data/Swiss/TreeAI_Swiss')
    folders = treeai_swiss_folder.iterdir()

    for folder in tqdm(folders, desc=f"Clipping height models"):
        dem_rasters = list(folder.glob("*swissalti3d*.tif"))
        dsm_rasters = list(folder.glob("*swisssurface3d*.tif"))

        assert len(dem_rasters) == len(dsm_rasters), "The length of DEM and DSM products should be the same."

        uav_raster = str(folder / (folder.name + '.tif'))
        raster_size_compare= []

        for height_model in ['dem', 'dsm']:
            clipped_height_model = str(folder / (folder.name + f'_{height_model}.tif'))
            height_model_rasters = dem_rasters if height_model == 'dem' else dsm_rasters

            if len(height_model_rasters) == 1:
                clip(input_raster=str(height_model_rasters[0]), clip_raster=uav_raster, output=clipped_height_model)
            else:
                mosaic_clip(input_rasters=height_model_rasters, clip_raster=uav_raster, output=clipped_height_model)

            raster_size = get_raster_size(clipped_height_model)
            raster_size_compare.append(raster_size)

            delete_files(height_model_rasters)
        if raster_size_compare[0] != raster_size_compare[1]:
            raise ValueError(f"Raster size of {folder.name} is different")



>>>>>>> e7e640b071f84c7a093ae1d39e9f5158e653688d
