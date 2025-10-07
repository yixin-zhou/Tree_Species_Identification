import calendar
import ee
from tqdm import tqdm

PROJECT_NAME = 'treeai-470815'
BUCKET_NAME = "treeai_swiss"
S2_BANDS = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12', 'SCL', 'QA60']

# Initialize the Earth Engine module and register this code in project 'TreeAI'
ee.Authenticate()
ee.Initialize(project=PROJECT_NAME)

swiss_boundary = "projects/treeai-470815/assets/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET"
swiss_boundary = ee.FeatureCollection(swiss_boundary)

for year in tqdm(range(2018, 2023), desc="Submitting Sentinel-2 composition tasks"):
    for month in range(1, 13):
        start_date = ee.Date.fromYMD(year, month, 1)
        end_date =  start_date.advance(1, "month")
        s2_sr = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                 .filterBounds(swiss_boundary.geometry())
                 .filterDate(start_date, end_date)
                 .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 10))
                 .select(S2_BANDS))

        # image_num = s2_sr.size().getInfo()
        # print(f"On {calendar.month_name[month]}, {year}, there are {image_num} images")

        swiss_s2 = s2_sr.median().clip(swiss_boundary)

        task = ee.batch.Export.image.toCloudStorage(
            image=swiss_s2,
            bucket=BUCKET_NAME,
            description=f"Sentinel-2_Swiss_{calendar.month_name[month]}_{year}",
            fileNamePrefix=f"Sentinel2/{year}_{calendar.month_name[month]}/Sentinel2_Swiss_{calendar.month_name[month]}_{year}_median",
            region=swiss_boundary.geometry(),
            crs="EPSG:2056",
            scale=10,
            maxPixels=1e13,
            fileFormat='GeoTIFF',

            formatOptions={'cloudOptimized': True}
        )

        task.start()

