import rioxarray as rxr
from osgeo import gdal
from pathlib import Path


def transCRS(input_raster, output_raster, dst_epsg_code=4326):
    input_raster = gdal.Open(input_raster)

    gdal.Warp(
        output_raster,
        input_raster,
        dstSRS=f'EPSG:{dst_epsg_code}',
        resampleAlg="bilinear"
    )

    input_raster = None


if __name__ == '__main__':
    images_dir = "../data/test"
    images = Path(images_dir).iterdir()
    i = 0
    for image in images:
        # transCRS(input_raster=str(image), output_raster=f"../data/{i}.tif")
        da = rxr.open_rasterio(str(image))  # 若没有 CRS，可先 da = da.rio.write_crs(2056)
        da4326 = da.rio.reproject("EPSG:4326")
        da4326.rio.to_raster(f"../data/{i}.tif")
        i += 1
