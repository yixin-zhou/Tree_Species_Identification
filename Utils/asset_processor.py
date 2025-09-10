import ee
import os
from tqdm import tqdm
import calendar
import json
from google.cloud import storage
from scripts.s1_preprocess.s1_ard import preprocess_s1
from scripts.s1_preprocess.helper import lin_to_db


class AssetProcessor:
    def __init__(self, asset_uri: str, year: int, bucket: str, exp_folder: dict):
        self.image = ee.Image(asset_uri)
        self.asset_name = os.path.basename(asset_uri)
        self.year = year

        self.ref_proj = self.image.projection()
        self.ref_crs = self.ref_proj.crs()
        self.aoi_rect = self.image.geometry().bounds(proj=self.ref_proj, maxError=1)

        self.default_bucket = bucket
        self.default_s2_bands = ['B2', 'B3', 'B4', 'B8',  # 10m
                                 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']  # 20m
        self.default_worldclim_scale_factor = [0.1, 0.1, 1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,
                                               0.1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.default_exp_folders = exp_folder
        self.default_maxPixels = 1e13
        self.default_exp_scale = 10
        self.s1_params_json = 'scripts/s1_preprocess/s1_preprocess_params.json'

    @staticmethod
    def count_assets(parent_path):
        info = ee.data.listAssets({'parent': parent_path})
        assets = info.get('assets', [])
        count = 0
        for asset in assets:
            count += 1
            if asset['type'] == 'FOLDER':
                count += AssetProcessor.count_assets(asset['id'])
        return count

    def _load_s1_preprocess_params(self, params_json, start_date, end_date, roi: ee.Geometry,
                                   dem=ee.Image('USGS/SRTMGL1_003')):
        with open(params_json, 'r', encoding='utf-8') as f:
            params = json.load(f)
        params['ROI'] = roi
        params['DEM'] = dem
        params['START_DATE'] = start_date
        params['STOP_DATE'] = end_date
        return params

    def _generate_monthly_composition(self, datasets: ee.ImageCollection):
        BANDS_FOR_PROCESSING = self.default_s2_bands + ['SCL']
        homogeneous_datasets = datasets.select(BANDS_FOR_PROCESSING)

        def mask_s2_clouds(image):
            scl = image.select('SCL')
            cloud_mask = scl.eq(3).Or(scl.eq(8)).Or(scl.eq(9)).Or(scl.eq(10))
            return image.updateMask(cloud_mask.Not())

        masked_images = homogeneous_datasets.map(mask_s2_clouds)
        monthly_composite_image = masked_images.median()
        return monthly_composite_image

    def _start_export_task(self, export_img, exp_folder, month=None, desc='', product=''):
        if month is None:
            filenameprefix = f'{exp_folder}/{self.asset_name}_{product}_{self.year}'
        else:
            filenameprefix = f'{exp_folder}/{self.asset_name}_{product}_{self.year}_{month}'

        task = ee.batch.Export.image.toCloudStorage(
            image=export_img,
            description=desc,
            bucket=self.default_bucket,
            fileNamePrefix=filenameprefix,
            region=self.aoi_rect,
            crs=self.ref_crs,
            scale=self.default_exp_scale,
            maxPixels=self.default_maxPixels,
            formatOptions={'cloudOptimized': True}
        )
        task.start()

    def export_SatEmbed(self):
        embed = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                 .filterDate(f'{self.year}-01-01', f'{self.year + 1}-01-01')
                 .filterBounds(self.aoi_rect)
                 .first()
                 .clip(self.aoi_rect))

        self._start_export_task(export_img=embed,
                                exp_folder=self.default_exp_folders['SatEmbed'],
                                desc=f'Satellite embedding of {self.asset_name}',
                                product='SatEmbed')

    def export_S1(self):
        for month in range(1, 13):
            start_date = ee.Date.fromYMD(self.year, month, 1)
            end_date = start_date.advance(1, 'month')
            params = self._load_s1_preprocess_params(self.s1_params_json,
                                                     start_date=start_date,
                                                     end_date=end_date,
                                                     roi=self.aoi_rect)
            s1_collections = preprocess_s1(params)
            if s1_collections.size().getInfo():
                monthly_s1 = s1_collections.mean()
                monthly_s1_db = lin_to_db(monthly_s1).select(['VV', 'VH'])
                reproject_s1 = monthly_s1_db.reproject(crs=self.ref_crs, scale=self.default_exp_scale)
                self._start_export_task(export_img=reproject_s1,
                                        exp_folder=self.default_exp_folders['S1'],
                                        desc=f'Sentinel-1 of {self.asset_name} on {calendar.month_name[month]}',
                                        product='S1')
            else:
                raise ValueError(f"There is no available Sentinel-1 images for {self.asset_name} in {self.year}")

    def export_S2(self):
        for month in range(1, 13):
            start_date = ee.Date.fromYMD(self.year, month, 1)
            end_date = start_date.advance(1, 'month')

            s2_collections = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterDate(start_date,
                                                                                          end_date).filterBounds(
                self.aoi_rect)
            if s2_collections.size().getInfo():
                monthly_image = self._generate_monthly_composition(s2_collections).select(self.default_s2_bands)
                resampled_monthly_image = monthly_image.reproject(
                    crs=self.ref_crs,
                    scale=self.default_exp_scale
                )
                self._start_export_task(export_img=resampled_monthly_image,
                                        exp_folder=self.default_exp_folders['S2'],
                                        desc=f'Sentinel-2 of {self.asset_name} on {calendar.month_name[month]}',
                                        product='S2')

    def exportClimateData(self, save_path):
        aoi_center = self.image.geometry().centroid(maxError=1)

        worldclim = ee.Image('WORLDCLIM/V1/BIO')
        band_names = [f'bio{i:02d}' for i in range(1, 20)]
        band_scale = dict(zip(band_names, self.default_worldclim_scale_factor))

        worldclim_proj = worldclim.projection()

        raw_climate_data = worldclim.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=aoi_center,
            crs=worldclim_proj,
            scale=worldclim_proj.nominalScale()
        ).getInfo()

        if None in raw_climate_data.values():
            raise ValueError(f"There is None data in the WorldClim data of {self.asset_name}. ")

        climate_data = {k: band_scale[k] * raw_climate_data[k] for k in band_scale}

        if save_path is not None:
            with open(save_path, "w", encoding='utf-8') as f:
                json.dump(climate_data, f, indent=4)

