import ee
import json
import calendar
from Utils.s1_preprocess.s1_ard import preprocess_s1
from Utils.s1_preprocess.helper import lin_to_db
import logging

PROJECT_NAME = 'treeai-465719'
BUCKET_NAME = "treeai_assets"

# Initialize the Earth Engine module and register this code in project 'TreeAI'
ee.Authenticate(force=True)
ee.Initialize(project=PROJECT_NAME)


def _load_s1_preprocess_params(params_json, start_date, end_date, roi: ee.Geometry,
                               dem=ee.Image('USGS/SRTMGL1_003')):
    with open(params_json, 'r', encoding='utf-8') as f:
        params = json.load(f)
    params['ROI'] = roi
    params['DEM'] = dem
    params['START_DATE'] = start_date
    params['STOP_DATE'] = end_date
    return params


def generate_monthly_Sentinel1(year, month, scale=10,
                               s1_params_json='Utils/s1_preprocess/s1_preprocess_params.json',
                               swiss_boundary="projects/treeai-465719/assets/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET"):
    swiss_boundary = ee.FeatureCollection(swiss_boundary)

    start_date = ee.Date.fromYMD(year, month, 1)
    end_date = start_date.advance(1, 'month')
    params = _load_s1_preprocess_params(params_json=s1_params_json,
                                        start_date=start_date,
                                        end_date=end_date,
                                        roi=swiss_boundary.geometry())
    s1_collections = preprocess_s1(params)

    if s1_collections.size().getInfo():
        monthly_s1 = s1_collections.median()
        monthly_s1_db = lin_to_db(monthly_s1).select(['VV', 'VH'])
        # reproject_s1 = monthly_s1_db.reproject(crs='EPSG:2056', scale=scale)

        monthly_s1_db = monthly_s1_db.clip(swiss_boundary)

        task = ee.batch.Export.image.toCloudStorage(
            image=monthly_s1_db,
            bucket=BUCKET_NAME,
            description=f"Sentinel-1 image for Switzerland in {calendar.month_name[month]}, {year}",
            fileNamePrefix=f"Sentinel-1/Sentinel1_Swiss_{calendar.month_name[month]}_{year}_median",
            region=swiss_boundary.geometry(),
            crs='EPSG:2056',
            scale=scale,
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}
        )

        task.start()
    else:
        logging.warning(
            f"There is no available Sentinel-1 images for Switzerland in {calendar.month_name[month]}, {year}")


if __name__ == '__main__':
    for year in range(2017, 2023):
        for month in range(1,4):
            generate_monthly_Sentinel1(year=year, month=month)
        for month in range(11, 13):
            generate_monthly_Sentinel1(year=year, month=month)

