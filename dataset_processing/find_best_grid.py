import geopandas as gpd
from sklearn.cluster import DBSCAN
import numpy as np
from shapely.geometry import box
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def snap_floor(v, cell, origin=0.0):
    return origin + np.floor((v - origin) / cell) * cell


def snap_ceil(v, cell, origin=0.0):
    return origin + np.ceil((v - origin) / cell) * cell


SEED = 42
CELL_SIZE = 60
EPS = 20
MIN_SAMPLES = 1
PAD_PERCENT = 0.1
GRID_ORIGIN = (0.0, 0.0)
annotations_shapefile = '../data/shapefile/annotations/Data_Swiss_NDVI_XY_50buf_up_28782_LF_MB_ZX_27790spDead.shp'
out_dir = '../data/shapefile/grid'

anno = gpd.read_file(annotations_shapefile)
anno_df = anno.explode(ignore_index=True)

# print(f"Sum of samples is {len(anno_df)}")
# print(anno.head())
# print(anno.columns)
# print(anno.geom_type.unique())
# print(anno.crs)

if {"XCoord", "YCoord"}.issubset(anno_df.columns):
    coords = anno_df[["XCoord", "YCoord"]].to_numpy(dtype=float)
else:
    raise KeyError("There is no column 'XCoord' and 'YCoord'")

db = DBSCAN(eps=EPS, min_samples=MIN_SAMPLES).fit(coords)
anno_df["cluster_id"] = db.labels_

origin_X = GRID_ORIGIN[0]
origin_Y = GRID_ORIGIN[1]

all_cells = []

for cid in np.unique(db.labels_):
    sub = anno_df[anno_df.cluster_id == cid]
    if sub.empty: continue

    minx, miny, maxx, maxy = sub.total_bounds  # get the coordinates of the bounding box

    pad = CELL_SIZE * PAD_PERCENT
    minx, miny = minx - pad, miny - pad
    maxx, maxy = maxx + pad, maxy + pad
    minx = snap_floor(minx, CELL_SIZE, origin_X)
    miny = snap_floor(miny, CELL_SIZE, origin_Y)
    maxx = snap_ceil(maxx, CELL_SIZE, origin_X)
    maxy = snap_ceil(maxy, CELL_SIZE, origin_Y)

    xs = np.arange(minx, maxx, CELL_SIZE)
    ys = np.arange(miny, maxy, CELL_SIZE)

    grid = gpd.GeoDataFrame(
        {"geometry": [box(x, y, x + CELL_SIZE, y + CELL_SIZE) for x in xs for y in ys]},
        crs=sub.crs
    )

    keep = gpd.sjoin(grid, sub[["geometry"]], how="inner", predicate="covers")
    keep = keep.drop(columns=[c for c in keep.columns if c not in ["geometry"]]).drop_duplicates()

    if not keep.empty:
        keep = keep.copy()
        keep["cluster_id"] = cid
        all_cells.append(keep)

fishnet = gpd.GeoDataFrame(pd.concat(all_cells, ignore_index=True), crs=anno_df.crs)
fishnet = fishnet.drop_duplicates(subset="geometry").reset_index(drop=True)
fishnet["grid_id"] = np.arange(0, len(fishnet), dtype=int)

out_fishnet = Path(out_dir) / "swiss_tree_fishnet_60m.shp"
fishnet.to_file(out_fishnet)

print(f"There are {len(fishnet)} grids left.")

# Spatial join the anno_df and fishnet
anno_df = anno_df.set_geometry("geometry")
fishnet = fishnet.set_geometry("geometry")
fishnet = fishnet.to_crs(anno_df.crs)

PREDICATE_SAMPLE_IN_GRID = "covered_by"
samples_with_grid = gpd.sjoin(
    anno_df.reset_index(drop=True).assign(sample_id=lambda d: np.arange(len(d))),
    fishnet[["grid_id", "geometry"]],
    how="left",
    predicate=PREDICATE_SAMPLE_IN_GRID
).drop(columns=["index_right"])

samples_with_grid = samples_with_grid.dropna(subset=["grid_id"]).reset_index(drop=True)

out_samples = Path(out_dir) / "swiss_tree_annotations_with_filtered_grid.shp"
samples_with_grid.to_file(out_samples)

cluster_to_grid = pd.Series(grid_to_cluster, name="cluster").groupby(level=0).first()
counts_c = counts.join(cluster_to_grid).groupby("cluster").sum()  # 行=cluster, 列=species

clusters = counts_c.index.tolist()
species  = counts_c.columns.tolist()