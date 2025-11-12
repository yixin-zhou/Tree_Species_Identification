import calendar
import ee
from tqdm import tqdm
import geopandas as gpd
from google.cloud import storage
from Utils.utils import getAcquireYear

# ---------------------------------------------------------------------------
# Constants and Earth Engine Initialization
# ---------------------------------------------------------------------------

PROJECT_NAME = 'treeai-470815'
BUCKET_NAME = "treeai_swiss"
OPTICAL_BANDS = [
    'B2', 'B3', 'B4', 'B5', 'B6',
    'B7', 'B8', 'B8A', 'B11', 'B12'
]


def scl_mask(img):
    # SCL: 0 No data, 1 Saturated/defective, 2 Dark, 3 Shadow, 4 Veg, 5 Bare, 6 Water,
    #      7 Unclassified, 8 Thin cirrus, 9 Cloud, 10 Cloud shadow, 11 Snow/Ice
    scl = img.select('SCL')
    good = scl.eq(4).Or(scl.eq(5)).Or(scl.eq(6))
    optical = img.select(OPTICAL_BANDS).divide(10000.0)
    masked = optical.updateMask(good)

    masked = masked.copyProperties(img, ['system:time_start'])

    return ee.Image(masked)


def generate_monthly_s2(year, month, aoi_rect, ref_proj):

    def compute_and_set_pixel_count(img):
        masked_optical = scl_mask(img)

        stats = masked_optical.select(OPTICAL_BANDS[0]).reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=aoi_rect,
            crs=ref_proj,
            scale=10,
            maxPixels=1e13
        )

        valid_pixels = stats.get(OPTICAL_BANDS[0])

        return masked_optical.set('valid_pixel_count', valid_pixels)

    start_date = ee.Date.fromYMD(int(year), month, 1)
    end_date = start_date.advance(1, 'month')
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(start_date, end_date)
           .filterBounds(aoi_rect)
           )
    a = col.size().getInfo()
    col_with_counts = col.map(compute_and_set_pixel_count)
    sorted_col = col_with_counts.sort('valid_pixel_count', False)
    best_image = ee.Image(sorted_col.first())

    month_prefix = calendar.month_name[month]
    renamed = best_image.rename([f"{month_prefix}_{b}" for b in OPTICAL_BANDS])
    return renamed


def generate_yearly_s2(year, aoi_rect):
    start_date = ee.Date.fromYMD(int(year), 1, 1)
    end_date = start_date.advance(1, 'year')
    col = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
           .filterDate(start_date, end_date)
           .filterBounds(aoi_rect)
           .map(scl_mask)
           .select(OPTICAL_BANDS)
           )
    yearly_image = col.median()
    renamed = yearly_image.rename([f"yearly_{b}" for b in OPTICAL_BANDS])
    return renamed


def check_missing_s2():
    target_grid = list(range(7340))
    storage_client = storage.Client()
    # bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix="Sentinel-2/"))
    s2_blobs = [blob for blob in blobs if blob.name.lower().endswith('.tif')]
    existed_grid = [int(s2_blob.name.split('/')[-1].replace('.tif', '')) for s2_blob in s2_blobs]
    missing_grid = set(target_grid) - set(existed_grid)
    return missing_grid


if __name__ == '__main__':
    # Initialize the Earth Engine module
    ee.Authenticate()
    ee.Initialize(project=PROJECT_NAME)

    # Swiss boundary FeatureCollection
    swiss_boundary = "projects/treeai-470815/assets/swissBOUNDARIES3D_1_5_TLM_LANDESGEBIET"
    swiss_boundary = ee.FeatureCollection(swiss_boundary)

    # Annotation shapefile
    ANNOTATIONS_SHP = "../data/shapefile/grid/swiss_tree_annotations_with_filtered_grid.shp"
    anno_df = gpd.read_file(ANNOTATIONS_SHP)

    # ---------------------------------------------------------------------------
    # Google Cloud Storage setup
    # ---------------------------------------------------------------------------

    storage_client = storage.Client()
    bucket = storage_client.bucket(BUCKET_NAME)
    blobs = list(storage_client.list_blobs(BUCKET_NAME, prefix="masks/"))
    raster_blobs = [blob for blob in blobs if blob.name.lower().endswith('.tif')]

    missing_grid = check_missing_s2()
    # for grid_id in tqdm(missing_grid, desc="Checking missing grid years"):
    #     year = getAcquireYear(grid_id, anno_df)
    #     if year != 2017:
    #         print(f"For grid {grid_id}, year is {year}")

    # ---------------------------------------------------------------------------
    # Main Loop
    # ---------------------------------------------------------------------------

    for blob in tqdm(raster_blobs, desc='Exporting Sentinel-2 data'):
        grid_id = int(blob.name.split('/')[-1][:4])

        if grid_id not in missing_grid:
            continue

        gcs_path = f"gs://{blob.bucket.name}/{blob.name}"
        fileNamePrefix = blob.name.replace('masks', 'Sentinel-2').replace('.tif', '')

        # Skip if Sentinel-2 export already exists
        s2_file = bucket.blob(fileNamePrefix + '.tif')
        if s2_file.exists():
            continue

        # Load corresponding UAV image for reference projection
        image = ee.Image.loadGeoTIFF(gcs_path)
        ref_proj = image.projection()
        ref_crs = ref_proj.crs()
        aoi_rect = image.geometry().bounds(proj=ref_proj, maxError=1)

        year = getAcquireYear(grid_id, anno_df)
        year = year + 1 if year == 2017 else year

        yearly_image = []

        # -----------------------------------------------------------------------
        # Generate monthly composites
        # -----------------------------------------------------------------------
        for month in range(1, 13):
            monthly_s2 = generate_monthly_s2(year, month, aoi_rect, ref_proj)
            yearly_image.append(monthly_s2)
        yearly_image.append(generate_yearly_s2(year, aoi_rect))
        # -----------------------------------------------------------------------
        # Concatenate monthly composites and export
        # -----------------------------------------------------------------------
        yearly_stack = ee.Image.cat(yearly_image)

        task = ee.batch.Export.image.toCloudStorage(
            image=yearly_stack,
            bucket=BUCKET_NAME,
            description=f"Sentinel-2 image for Grid {grid_id}",
            fileNamePrefix=fileNamePrefix,
            region=aoi_rect,
            crs='EPSG:2056',
            scale=10,
            maxPixels=1e13,
            fileFormat='GeoTIFF',
            formatOptions={'cloudOptimized': True}
        )

        task.start()
