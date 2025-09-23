import rasterio
from pathlib import Path
import numpy as np
from scipy.ndimage import label, find_objects
from tqdm import tqdm

MASK_SIZE = (500, 500)

def numpy2bbox(nd_array, background=0):
    bboxes = []
    label_types = np.unique(nd_array)
    label_types = label_types[label_types != background]
    for label_type in label_types:
        binary_mask = (nd_array == label_type).astype(np.uint8)
        labeled_array, num_features = label(binary_mask, structure=np.ones((3, 3)))
        slices = find_objects(labeled_array)
        for s in slices:
            r, c = s
            xmin, xmax = c.start, c.stop
            ymin, ymax = r.start, r.stop
            box = {'bbox': [xmin, ymin, xmax - xmin, ymax - ymin], 'category': label_type}
            bboxes.append(box)
    return bboxes


def bbox2txt(bboxes, output_path, img_size=MASK_SIZE):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img_w, img_h = img_size
    with open(output_path, 'w') as f:
        for box in bboxes:
            xmin, ymin, w, h = box['bbox']
            category = box['category']
            x_center = (xmin + w / 2) / img_w
            y_center = (ymin + h / 2) / img_h
            norm_w = w / img_w
            norm_h = h / img_h
            f.write(f"{category} {x_center} {y_center} {norm_w} {norm_h}\n")



masks_path = '../data/TreeAI_Swiss/masks'
masks = list(Path(masks_path).rglob("*.tif"))
for mask in tqdm(masks, desc='Transferring masks to bouding boxes'):
    txt_path = Path(str(mask.parent).replace('masks','labels')) / mask.name.replace('.tif', '.txt')
    with rasterio.open(mask) as f:
        mask_array = f.read(1)
        bboxes = numpy2bbox(mask_array)
        bbox2txt(bboxes, txt_path)

