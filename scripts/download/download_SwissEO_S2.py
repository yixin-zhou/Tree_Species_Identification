import ee
import os
from tqdm import tqdm
import calendar
import json
from google.cloud import storage
import time

default_s2_bands = ['B2', 'B3', 'B4', 'B8',  # 10m
                    'B5', 'B8A', 'B11']  # 20m

bands10 = ['B2','B3','B4','B8']
bands20 = ['B5','B8A','B11']

def maskBySCL(s2_image, masked_type=[0, 1, 3, 7, 8, 9, 10, 11]):
    scl = s2_image.select('SCL').resample('nearest')
    img_bilinear = s2_image.resample('bilinear')
    img_bilinear = img_bilinear.addBands(scl, overwrite=True)

    bad_values = ee.List(masked_type)
    bad_flag = scl.remap(bad_values, ee.List.repeat(1, bad_values.length()), 0)

    good_mask = bad_flag.neq(1)

    masked_img = img_bilinear.updateMask(good_mask)
    return masked_img


# def combine_pair(i):
#     i = ee.Number(i)
#     i10 = ee.Image(l10.get(i))
#     i20 = ee.Image(l20.get(i))
#     return (ee.Image.cat([
#                 i10.select(['B2', 'B3', 'B4', 'B8', 'cloudAndCloudShadowMask']),
#                 i20.select(['B5', 'B8A', 'B11'])
#         ]).copyProperties(i10, ['system:time_start', 'system:index']))
#
#
# def upsample20mBands(img):
#     ref10 = img.select('B2').projection()
#
#     bands20_to10 = (img.select(['B5', 'B8A', 'B11'])
#                     .resample('bilinear')
#                     .reproject(ref10))
#     bands10 = img.select(['B2', 'B3', 'B4', 'B8'])
#     mask10 = img.select('cloudAndCloudShadowMask')
#     out = (ee.Image.cat([bands10, bands20_to10, mask10])
#             .updateMask(mask10.eq(0))
#             .copyProperties(img, img.propertyNames()))
#     return out

PROJECT_NAME = 'treeai-470815'
EXP_SCALE = 10
MAX_PIXELS = 1e13

# Initialize the Earth Engine module and register this code in project 'TreeAI'
ee.Authenticate()
ee.Initialize(project=PROJECT_NAME)

swiss_boundary = ee.FeatureCollection("projects/treeai-470815/assets/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET")

if __name__ == '__main__':
    year = 2018
    for month in range(1, 13):
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')

        swiss_S2_images = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
                            .filterBounds(swiss_boundary.geometry()) \
                            .filterDate(start_date, end_date) \
                            .map(lambda im: maskBySCL(im))

        comp10 = swiss_S2_images.select(bands10).median().clip(swiss_boundary)
        comp20_native = swiss_S2_images.select(bands20).median().clip(swiss_boundary)

        comp20_at10m = comp20_native.resample('bilinear').reproject(comp10.projection())
        S2_10m = comp10.addBands(comp20_at10m).set({
            'year': year, 'month': month, 'strategy': 'SCL_median', 'scale': 10
        })

        ee.batch.Export.image.toCloudStorage(
            image=S2_10m,
            description=f"S2_CH_{year}_{month:02d}_median_SCL_10m_EPSG2056",
            bucket='your-bucket-name',
            fileNamePrefix=f"Sentinel-2/S2_CH_{year}_{month:02d}_median_SCL_10m_EPSG2056",
            region=swiss_boundary.geometry(),
            scale=10,
            crs='EPSG:2056',
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}
        ).start()


# orbits = [65, 108]
#
# orbit_65_sum = 0
# orbit_108_sum = 0
# for year in range(2019, 2023):
#     for month in range(1, 13):
#         start_date = ee.Date.fromYMD(year, month, 1)
#         end_date = start_date.advance(1, 'month')
#
#         orbit_image_num = []
#         for orbit in orbits:
#             collections_108 = ee.ImageCollection("projects/satromo-prod/assets/col/S2_SR_HARMONIZED_SWISS") \
#                                 .filterDate(start_date, end_date) \
#                                 .filterMetadata('SENSING_ORBIT_NUMBER', 'equals', orbit)
#             images_num = collections_108.size().getInfo()
#             if orbit == 65:
#                 orbit_65_sum += 1
#             else:
#                 orbit_108_sum += 1
#             orbit_image_num.append(images_num)
#
#         print(f"On {calendar.month_name[month]}, {year}, {orbit_image_num[0]} images for Orbit 65, {orbit_image_num[1]} images for Orbit 108")
#
# print(f"From 2019-2022, Orbit 65 has {orbit_65_sum} available images, Orbit 108 has {orbit_108_sum} available images")
