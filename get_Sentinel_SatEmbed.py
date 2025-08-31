import ee
import os
from tqdm import tqdm
import calendar
import json
# from scripts.s1_preprocess.s1_ard import preprocess_s1

S2_BANDS = ['B2', 'B3', 'B4', 'B8',  # 10m
            'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']  # 20m

# Initialize the Earth Engine module and register this code in project 'TreeAI'
ee.Authenticate()
ee.Initialize(project='treeai-465719')


def load_s1_preprocess_params(params_json, start_date, end_date, roi: ee.Geometry,
                              dem=ee.Image('USGS/SRTMGL1_003')):
    with open(params_json, 'r', encoding='utf-8') as f:
        params = json.load(f)
    params['ROI'] = roi
    params['DEM'] = dem
    params['START_DATE'] = start_date
    params['STOP_DATE'] = end_date
    return params


def count_assets(parent_path):
    info = ee.data.listAssets({'parent': parent_path})
    assets = info.get('assets', [])
    count = 0
    for asset in assets:
        count += 1
        if asset['type'] == 'FOLDER':
            count += count_assets(asset['id'])
    return count


def exportSatEmbed(asset_id, year, exp_scale=10, exp_folder='gee_export_Sat_Embd', maxPixels=1e13):
    img = ee.Image(asset_id)
    asset_name = os.path.basename(asset_id)

    # Get projection information of src UAV image
    ref_proj = img.projection()
    ref_crs = ref_proj.crs()
    aoi_rect = img.geometry().bounds(proj=ref_proj, maxError=1)

    # find corresponding satellite embedding dataset v1 image
    emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
           .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
           .filterBounds(aoi_rect)
           .first()
           .clip(aoi_rect))

    # define the task parameters and submit the task to Google Cloud
    task = ee.batch.Export.image.toDrive(
        image=emb,
        description=f'Satellite embedding of {asset_name}',
        folder=exp_folder,
        fileNamePrefix=f'{asset_name}_SatEmbd_{year}',
        region=aoi_rect,
        crs=ref_crs,
        scale=exp_scale,
        maxPixels=maxPixels,
        formatOptions={'cloudOptimized': True}
    )
    task.start()


def generate_monthly_composition(datasets: ee.ImageCollection, S2_bands=S2_BANDS):
    BANDS_FOR_PROCESSING = S2_bands + ['SCL']
    homogeneous_datasets = datasets.select(BANDS_FOR_PROCESSING)

    def mask_s2_clouds(image):
        scl = image.select('SCL')
        cloud_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
        return image.updateMask(cloud_mask.Not())

    masked_images = homogeneous_datasets.map(mask_s2_clouds)
    monthly_composite_image = masked_images.median()
    return monthly_composite_image


def exportSentinel2(asset_id, year, exp_scale=10, exp_folder='gee_export_Sentinel2', maxPixels=1e13,
                    target_bands=S2_BANDS):
    img = ee.Image(asset_id)
    asset_name = os.path.basename(asset_id)

    # Get projection information of src UAV image
    ref_proj = img.projection()
    ref_crs = ref_proj.crs()
    aoi_rect = img.geometry().bounds(proj=ref_proj, maxError=1)

    for month in range(1, 13):
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')

        dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(start_date, end_date).filterBounds(
            aoi_rect)
        image_num = dataset.size().getInfo()
        print(f"For {asset_name}, find {image_num} available images in {calendar.month_name[month]}")

        # if image_num == 0:
        #     raise ValueError(
        #         f"There is no available Sentinel-2 images for {asset_name} in {calendar.month_name[month]}.")

        # monthly_image = generate_monthly_composition(dataset).select(target_bands)
        # resampled_monthly_image = monthly_image.reproject(
        #     crs=ref_crs,
        #     scale=exp_scale
        # )
        #
        # task = ee.batch.Export.image.toDrive(
        #     image=resampled_monthly_image,
        #     description=f'Sentinel-2 of {asset_name} on {calendar.month_name[month]}',
        #     fileNamePrefix=f'{asset_name}_S2_{year}_{month}',
        #     folder=exp_folder,
        #     scale=exp_scale,
        #     crs=ref_crs,
        #     region=aoi_rect,
        #     maxPixels=maxPixels,
        #     fileFormat='GeoTIFF',
        #     formatOptions={'cloudOptimized': True}
        # )

        # task.start()


def export_Sentinel1(asset_id, year, exp_folder='gee_export_Sentinel1', maxPixels=1e13,
                     params_json='scripts/s1_preprocess/s1_preprocess_params.json'):
    img = ee.Image(asset_id)
    asset_name = os.path.basename(asset_id)

    # Get projection information of src UAV image
    ref_proj = img.projection()
    ref_crs = ref_proj.crs()
    aoi_rect = img.geometry().bounds(proj=ref_proj, maxError=1)

    for month in range(1, 13):
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date = start_date.advance(1, 'month')
        params = load_s1_preprocess_params(params_json,
                                           start_date=start_date,
                                           end_date=end_date,
                                           roi=aoi_rect)
        s1_collections = preprocess_s1(params)




if __name__ == '__main__':
    # Check if the upload is successful, if so, the number of shapefile assets should be the
    # same as the number of UAV images.
    gee_proj_root = 'users/bryce001006'
    total = count_assets(gee_proj_root)
    print(f"There are {total} assets under {gee_proj_root}")

    assets_list = ee.data.listAssets(gee_proj_root)['assets']

    for asset in tqdm(assets_list, desc=f'Downloading Satellite Embedding Dataset V1 and Sentinel-2'):
        year = int(asset['name'].split('/')[-1][:4])
        # year = year + 1 if year == 2017 else year
        # exportSatEmbed(asset_id=asset['name'], year=year)
        exportSentinel2(asset_id=asset['name'], year=year)
        # export_Sentinel1(asset_id=asset['name'], year=year)
