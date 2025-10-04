import rasterio
from pathlib import Path
from urllib.parse import urljoin
from rasterio.warp import transform
from tqdm import tqdm

CHELSA_base_URL = "https://os.zhdk.cloud.switch.ch/chelsav2/GLOBAL/climatologies/1981-2010/bio/"
CHELSA_offset = [-273.15, 0, 0, 0, -273.15, -273.15, 0, -273.15, -273.15, -273.15, -273.15, 0, 0, 0, 0, 0 ,0 ,0, 0]

images_folder = Path("data/TreeAI_Swiss/images")
uav_images = list(images_folder.rglob("*.tif"))

for uav_image in tqdm(uav_images, desc='Getting BIO values of CHELSA dataset'):
    with rasterio.open(uav_image) as src:
        epsg_code = src.crs.to_epsg()
        assert epsg_code == 2056, "The CRS of input .tif raster should be EPSG:2056"
        xmin, ymin, xmax, ymax = src.bounds
        cx = (xmin+xmax)/2
        cy = (ymin+ymax)/2

        lon, lat = transform(src.crs, "EPSG:4326", [cx], [cy])
        lon, lat = lon[0], lat[0]

    chelsa_values = []
    for bio_num in range(1,20):
        CHELSA_filename = f"CHELSA_bio{bio_num}_1981-2010_V.2.1.tif"
        CHELSA_URL = urljoin(CHELSA_base_URL, CHELSA_filename)
        with rasterio.open(CHELSA_URL) as chelsa:
            chelsa_epsg = chelsa.crs.to_epsg()
            assert chelsa_epsg == 4326, "The CRS of input .tif raster should be WGS84"
            
            bio_value = list(chelsa.sample([(lon, lat)]))[0][0] * 0.1 + CHELSA_offset[bio_num-1]
            chelsa_values.append(bio_value)
    
    save_path = Path(str(uav_image).replace('images','climate').replace('.tif','.txt'))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(" ".join(str(val) for val in chelsa_values))
