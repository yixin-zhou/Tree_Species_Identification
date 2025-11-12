import rasterio
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

def _tif2png(tif_path, png_path=None, clip_percent: tuple = (0, 100)):
    tif_path = Path(tif_path)
    if png_path is None:
        png_path = tif_path.with_suffix(".png")
    else:
        png_path = Path(png_path)

    Path(png_path.parent).mkdir(parents=True, exist_ok=True)

    with rasterio.open(tif_path) as src:
        n_bands = src.count
        if n_bands > 3:
            img = np.stack([src.read(b) for b in [1, 2, 3]], axis=-1).astype(np.float32)

    low, high = np.percentile(img, clip_percent)
    img = np.clip((img - low) / (high - low), 0, 1)
    img_uint8 = (img * 255).astype(np.uint8)
    plt.imsave(png_path, img_uint8)


if __name__ == '__main__':
    images = list(Path('../data/TreeAI_Swiss_60/images').rglob("*.tif"))
    for image in tqdm(images):
        png_path = Path(str(image).replace('images', 'images_png')).with_suffix('.png')
        _tif2png(image, png_path=png_path)
