import ee
import os
from tqdm import tqdm
import calendar
import json
from google.cloud import storage
import time


PROJECT_NAME = 'treeai-470815'
BUCKET_NAME = 'treeai_swiss'
EXP_SCALE = 10
MAX_PIXELS = 1e13
PARALLEL_TASKS = 299

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
ee.Authenticate()
ee.Initialize(project=PROJECT_NAME)

storage_client = storage.Client()
bucket = storage_client.bucket(BUCKET_NAME)
blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix="images/"))
raster_blobs = [blob for blob in blobs if blob.name.lower().endswith('.tif')]


# print(len(raster_blobs))

submitted_tasks = 0

for blob in tqdm(raster_blobs, desc='Exporting Satellite Embedding data'):
    gcs_path = f"gs://{blob.bucket.name}/{blob.name}"
        
    filename = f"{blob.name.split('/')[1]}_{blob.name.split('/')[2]}.tif"
    fileNamePrefix = blob.name.replace('images','Satellite_Embedding').replace('.tif', '')
    
    # Check if this file existed
    sat_embed_file = bucket.blob(fileNamePrefix + '.tif')
    if sat_embed_file.exists():
        continue
    
    image = ee.Image.loadGeoTIFF(gcs_path)
    
    # Get projection information of src UAV image
    ref_proj = image.projection()
    ref_crs = ref_proj.crs()
    aoi_rect = image.geometry().bounds(proj=ref_proj, maxError=1)
    
    year = int(blob.name.split('/')[1][:4])
    
    # find corresponding Satellite Embedding Dataset v1 image
    emb = (ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
           .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
           .filterBounds(aoi_rect)
           .first()
           .clip(aoi_rect))
    
    task = ee.batch.Export.image.toCloudStorage(
        image=emb,
        bucket=BUCKET_NAME,
        description=filename,
        fileNamePrefix=fileNamePrefix,
        region=aoi_rect,
        crs=ref_crs,
        scale=EXP_SCALE,
        maxPixels=MAX_PIXELS,
        formatOptions={'cloudOptimized': True}
    )
    
    task.start()
    
    submitted_tasks += 1
    if submitted_tasks == PARALLEL_TASKS:
        wait()
        submitted_tasks = 0