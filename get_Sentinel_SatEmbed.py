import ee
import os
from tqdm import tqdm
import calendar
import json
from google.cloud import storage
from Utils.s1_preprocess.s1_ard import preprocess_s1
from Utils.s1_preprocess.helper import lin_to_db

PROJECT_NAME = 'treeai-470815'
BUCKET_NAME = 'treeai_data'
S2_BANDS = ['B2', 'B3', 'B4', 'B8',  # 10m
            'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']  # 20m
worldclim_scale_factor = [0.1, 0.1, 1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1]

# Initialize the Earth Engine module and register this code in project 'TreeAI'
ee.Authenticate()
ee.Initialize(project=PROJECT_NAME)


class FindExportGEEProducts:
    def __init__(self, asset_uri):
        self.asset_uri = asset_uri



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


def exportSatEmbed(asset_id, year, exp_scale=10, exp_folder='Sat_Embed', maxPixels=1e13, bucket=BUCKET_NAME):
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
    task = ee.batch.Export.image.toCloudStorage(
        image=emb,
        description=f'Satellite embedding of {asset_name}',
        bucket=bucket,
        fileNamePrefix=f'{exp_folder}/{asset_name}_SatEmbed_{year}',
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

        dataset = ee.ImageCollection("projects/satromo-prod/assets/col/S2_SR_HARMONIZED_SWISS").filterDate(start_date, end_date).filterBounds(
            aoi_rect)
        image_num = dataset.size().getInfo()
        print(f"For {asset_name}, find {image_num} available images in {calendar.month_name[month]}")

        if image_num == 0:
            raise ValueError(
                f"There is no available Sentinel-2 images for {asset_name} in {calendar.month_name[month]}.")

        monthly_image = generate_monthly_composition(dataset).select(target_bands)
        resampled_monthly_image = monthly_image.reproject(
            crs=ref_crs,
            scale=exp_scale
        )

        task = ee.batch.Export.image.toDrive(
            image=resampled_monthly_image,
            description=f'Sentinel-2 of {asset_name} on {calendar.month_name[month]}',
            fileNamePrefix=f'{asset_name}_S2_{year}_{month}',
            folder=exp_folder,
            scale=exp_scale,
            crs=ref_crs,
            region=aoi_rect,
            maxPixels=maxPixels,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}
        )

        task.start()


def export_Sentinel1(asset_uri, year, exp_folder='gee_export_Sentinel1', maxPixels=1e13, exp_scale=10,
                     params_json='scripts/s1_preprocess/s1_preprocess_params.json', bucket_name=BUCKET_NAME):
    img = ee.Image.loadGeoTIFF(asset_uri)
    asset_name = os.path.basename(asset_uri)

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
        if s1_collections.size().getInfo():
            monthly_s1 = s1_collections.mean()
            monthly_s1_db = lin_to_db(monthly_s1).select(['VV', 'VH'])
            reproject_s1 = monthly_s1_db.reproject(crs=ref_crs, scale=exp_scale)

            task = ee.batch.Export.image.toCloudStorage(
                image=reproject_s1,
                description=f'Sentinel-1 of {asset_name} on {calendar.month_name[month]}',
                bucket=bucket_name,
                fileNamePrefix=f'{exp_folder}/{asset_name}_S1_{year}_{month}',
                region=aoi_rect,
                scale=exp_scale,
                crs=ref_crs,
                maxPixels=maxPixels,
                fileFormat='GeoTIFF',
                formatOptions={'cloudOptimized': True}
            )
            task.start()
        else:
            raise ValueError(f"There is no available Sentinel-1 images for {asset_name} in {year}")



def exportClimateData(asset_uri, save_path, band_scale_factor=worldclim_scale_factor):
    img = ee.Image.loadGeoTIFF(asset_uri)
    asset_name = os.path.basename(asset_uri)
    aoi_center = img.geometry().centroid(maxError=1)

    worldclim = ee.Image('WORLDCLIM/V1/BIO')
    band_names = [f'bio{i:02d}' for i in range(1, 20)]
    band_scale = dict(zip(band_names, band_scale_factor))

    worldclim_proj = worldclim.projection()

    raw_climate_data = worldclim.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=aoi_center,
        crs=worldclim_proj,
        scale=worldclim_proj.nominalScale()
    ).getInfo()

    if None in raw_climate_data.values():
        raise ValueError(f"There is None data in the WorldClim data of {asset_name}. ")

    climate_data = {k: band_scale[k] * raw_climate_data[k] for k in band_scale}

    if save_path is not None:
        with open(save_path, "w", encoding='utf-8') as f:
            json.dump(climate_data, f, indent=4)


if __name__ == '__main__':
    storage_client = storage.Client()
    blobs = list(storage_client.list_blobs(BUCKET_NAME))
    raster_blobs = [blob for blob in blobs if blob.name.lower().endswith('.tif')]

    print(f"There are {len(raster_blobs)} raster images under {PROJECT_NAME}/{BUCKET_NAME}")

    # assets_list = ee.data.listAssets(gee_proj_root)['assets']
    print("Begin exporting Sentinel-1, Sentinel-2, Satellite Embedding and WorldClim data.")
    for blob in tqdm(raster_blobs, desc="Exporting data"):
        year = int(blob.name.split('/')[-1][:4])
        gcs_uri = f'gs://{blob.bucket.name}/{blob.name}'

        exportClimateData(asset_uri=gcs_uri, save_path=None)
        export_Sentinel1(asset_uri=gcs_uri, year=year)
        # exportSentinel2(asset_id=asset['name'], year=Sentinel2_year)
        # exportSatEmbed(asset_id=asset['name'], year=year)
