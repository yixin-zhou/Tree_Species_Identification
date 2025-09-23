import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def yolo2coco(x_center, y_center, norm_w, norm_h, img_size=(500, 500)):
    img_w, img_h = img_size
    w = norm_w * img_w
    h = norm_h * img_h
    x_center = x_center * img_w
    y_center = y_center * img_h
    xmin = x_center - w / 2
    ymin = y_center - h / 2
    return [xmin, ymin, w, h]

def visualize_mask_with_bbox(mask_path, label_path, img_size=(500, 500)):
    with rasterio.open(mask_path) as f:
        mask = f.read(1)
    labels = np.loadtxt(label_path)
    labels = np.atleast_2d(labels)

    plt.figure(figsize=(10, 10))
    plt.imshow(mask)  # 直接用一个离散色表
    plt.axis("off")
    plt.tight_layout()

    for label in labels:
        class_id = int(label[0])
        x_center, y_center, w, h = label[1:].astype(float)
        xmin, ymin, bw, bh = yolo2coco(x_center, y_center, w, h, img_size=img_size)
        rect = patches.Rectangle(
            (xmin, ymin), bw, bh,
            linewidth=2, edgecolor="red", facecolor="none"
        )
        plt.gca().add_patch(rect)
    plt.show()

visualize_mask_with_bbox('../data/TreeAI_Swiss/masks/20170608_0930_12501_0_33/1.tif',
                         '../data/TreeAI_Swiss/labels/20170608_0930_12501_0_33/1.txt')





