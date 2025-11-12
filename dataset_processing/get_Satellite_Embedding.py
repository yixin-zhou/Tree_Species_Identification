import ee
from tqdm import tqdm
from google.cloud import storage
import time
import geopandas as gpd
from Utils.utils import getAcquireYear


PROJECT_NAME = 'treeai-470815'
BUCKET_NAME = 'treeai_swiss'
EXP_SCALE = 10
MAX_PIXELS = 1e13
PARALLEL_TASKS = 2999


def wait(check_internal=100):
    print("Stop submitting tasks, waiting for all submitted tasks are completed")
    while True:
        ops = ee.data.listOperations()
        pending = sum(1 for o in ops if o.get("metadata", {}).get("state") == "PENDING")
        if pending == 0:
            break
        else:
            print(f"Stop submitting tasks, {pending} tasks are still pending...")
            time.sleep(check_internal)


# Initialize the Earth Engine module and register this code in project 'TreeAI'
ee.Authenticate(force=True)
ee.Initialize(project=PROJECT_NAME)

ANNOTATIONS_SHP = "../data/shapefile/grid/swiss_tree_annotations_with_filtered_grid.shp"
anno_df = gpd.read_file(ANNOTATIONS_SHP)

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix="masks/"))
raster_blobs = [blob for blob in blobs if blob.name.lower().endswith('.tif')]

# print(len(raster_blobs))

submitted_tasks = 0

for blob in tqdm(raster_blobs, desc='Exporting Satellite Embedding data'):
    gcs_path = f"gs://{blob.bucket.name}/{blob.name}"

    fileNamePrefix = blob.name.replace('masks','Satellite_Embedding').replace('.tif', '')

    # Check if this file existed
    sat_embed_file = bucket.blob(fileNamePrefix + '.tif')
    if sat_embed_file.exists():
        continue

    image = ee.Image.loadGeoTIFF(gcs_path)

    # Get projection information of src UAV image
    ref_proj = image.projection()
    ref_crs = ref_proj.crs()
    aoi_rect = image.geometry().bounds(proj=ref_proj, maxError=1)

    grid_id = int(blob.name.split('/')[-1][:4])
    year = getAcquireYear(grid_id, anno_df)

    # find corresponding Satellite Embedding Dataset v1 image
    emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
           .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
           .filterBounds(aoi_rect)
           .first()
           .clip(aoi_rect))

    task = ee.batch.Export.image.toCloudStorage(
        image=emb,
        bucket=BUCKET_NAME,
        description=f"Satellite Embedding for Grid {grid_id}",
        fileNamePrefix=fileNamePrefix,
        region=aoi_rect,
        crs=ref_crs,
        scale=EXP_SCALE,
        maxPixels=MAX_PIXELS,
        formatOptions={'cloudOptimized': True}
    )

    task.start()
    #
    # submitted_tasks += 1
    # if submitted_tasks == PARALLEL_TASKS:
    #     wait()
    #     submitted_tasks = 0



# # Initialize the Earth Engine module
# ee.Authenticate(force=True)
# ee.Initialize(project=PROJECT_NAME)
#
# swiss_boundary = "projects/treeai-470815/assets/swissBOUNDARIES3D_1_5_TLM_KANTONSGEBIET"
# grid_buffer = "projects/treeai-470815/assets/swiss_tree_fishnet_60m_buffer"
# swiss_boundary = ee.FeatureCollection(swiss_boundary)
# grid_buffer = ee.FeatureCollection(grid_buffer)
#
# for year in tqdm(range(2017, 2023), desc="Submitting Satellite Embedding tasks"):
#     start_date = ee.Date.fromYMD(year, 1, 1)
#     end_date = start_date.advance(1, 'year')
#
#     sate_embed = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
#                   .filterBounds(swiss_boundary.geometry())
#                   .filterDate(start_date, end_date)
#                   .first())
#
#     # sate_embed = sate_embed.resample('bilinear')
#
#     task = ee.batch.Export.image.toCloudStorage(
#         image=sate_embed,
#         bucket=BUCKET_NAME,
#         description=f"Satellite Embedding for Switzerland in {year}",
#         fileNamePrefix=f"Satellite_Embedding/Satellite_Embedding_{year}",
#         region=grid_buffer.geometry(),
#         crs='EPSG:2056',
#         scale=10,
#         maxPixels=1e13,
#         fileFormat='GeoTIFF',
#         formatOptions={'cloudOptimized': True}
#     )
#
#     task.start()
