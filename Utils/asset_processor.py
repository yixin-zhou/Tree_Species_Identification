import ee
import os
import logging
import json
from Utils.s1_preprocess.s1_ard import preprocess_s1
from Utils.s1_preprocess.helper import lin_to_db


class AssetProcessor:
    def __init__(self, asset_uri: str, year: int, bucket: str, exp_folder: dict):
        self.image = ee.Image(asset_uri)
        # self.image = ee.Image.loadGeoTIFF(asset_uri)
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
        self.s1_params_json = 'Utils/s1_preprocess/s1_preprocess_params.json'

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


    def generate_SatEmbed(self):
        embed = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
                 .filterDate(f'{self.year}-01-01', f'{self.year + 1}-01-01')
                 .filterBounds(self.aoi_rect)
                 .first()
                 .clip(self.aoi_rect))
        return embed


    def generate_monthly_Sentinel1(self):
        s1_ts = []
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
                s1_ts.append(reproject_s1)
            else:
                logging.warning(f"There is no available Sentinel-1 images for {self.asset_name} in {self.year}")
        return s1_ts

    def generate_monthly_SwissEO(self):
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

        s2_ts = []

        for month in range(1, 13):
            start_date = ee.Date.fromYMD(self.year, month, 1)
            end_date = start_date.advance(1, 'month')

            col = (ee.ImageCollection("projects/satromo-prod/assets/col/S2_SR_HARMONIZED_SWISS")
                   .filterBounds(self.image.geometry())
                   .filterDate(start_date, end_date))

            images_10m = col.filter(ee.Filter.eq('pixel_size_meter', 10)).sort('system:time_start', True)
            images_20m = col.filter(ee.Filter.eq('pixel_size_meter', 20)).sort('system:time_start', True)

            n = images_10m.size()
            if n.getInfo()>0:
                l10 = images_10m.toList(n)
                l20 = images_20m.toList(n)

                merged = ee.ImageCollection(ee.List.sequence(0, n.subtract(1)).map(combine_pair))
                merged = merged.map(upsample20mBands)
                monthly_median = merged.select(self.default_s2_bands).median().clip(self.aoi_rect)
                s2_ts.append(monthly_median)
            else:
                logging.warning(f"There is no available Sentinel-2 images for {self.asset_name} in {self.year}")
        return s2_ts

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


    def mergeAllSatelliteData(self):
        sat_embed = self.generate_SatEmbed()
        s1_ts = ee.ImageCollection(self.generate_monthly_Sentinel1())
        s2_ts = ee.ImageCollection(self.generate_monthly_SwissEO())

        # Rename the band names first, no need for Satellite Embedding data
        return 0
            




    def exportAllSatelliteData(self):
        
        return satembed, s1_ts, s2_ts
