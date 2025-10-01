import ee
import os
from tqdm import tqdm
import calendar
import json
from google.cloud import storage
import time


default_s2_bands = ['B2', 'B3', 'B4', 'B8',  # 10m
                    'B5', 'B8A', 'B11']  # 20m


def combine_pair(i):
    i = ee.Number(i)
    i10 = ee.Image(l10.get(i))
    i20 = ee.Image(l20.get(i))
    return (ee.Image.cat([
                i10.select(['B2', 'B3', 'B4', 'B8', 'cloudAndCloudShadowMask']),
                i20.select(['B5', 'B8A', 'B11'])
        ]).copyProperties(i10, ['system:time_start', 'system:index']))


def upsample20mBands(img):
    ref10 = img.select('B2').projection()

    bands20_to10 = (img.select(['B5', 'B8A', 'B11'])
                    .resample('bilinear')
                    .reproject(ref10))
    bands10 = img.select(['B2', 'B3', 'B4', 'B8'])
    mask10 = img.select('cloudAndCloudShadowMask')
    out = (ee.Image.cat([bands10, bands20_to10, mask10])
            .updateMask(mask10.eq(0))
            .copyProperties(img, img.propertyNames()))
    return out

PROJECT_NAME = 'treeai-470815'
EXP_SCALE = 10
MAX_PIXELS = 1e13

# Initialize the Earth Engine module and register this code in project 'TreeAI'
ee.Authenticate()
ee.Initialize(project=PROJECT_NAME)

swiss_boundary = ee.FeatureCollection("projects/treeai-470815/assets/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET")

orbits = [65, 108]

orbit_65_sum = 0
orbit_108_sum = 0
for year in range(2019, 2023):
    for month in range(1, 13):
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
        
        orbit_image_num = []
        for orbit in orbits:
            collections_108 = ee.ImageCollection("projects/satromo-prod/assets/col/S2_SR_HARMONIZED_SWISS") \
                                .filterDate(start_date, end_date) \
                                .filterMetadata('SENSING_ORBIT_NUMBER', 'equals', orbit)
            images_num = collections_108.size().getInfo()
            if orbit == 65:
                orbit_65_sum += 1
            else:
                orbit_108_sum += 1
            orbit_image_num.append(images_num)
        
        print(f"On {calendar.month_name[month]}, {year}, {orbit_image_num[0]} images for Orbit 65, {orbit_image_num[1]} images for Orbit 108")

print(f"From 2019-2022, Orbit 65 has {orbit_65_sum} available images, Orbit 108 has {orbit_108_sum} available images")
