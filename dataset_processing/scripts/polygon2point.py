import geopandas as gpd
import pandas as pd

annotations_shapefile = "../data/shapefile/grid/swiss_tree_annotations_with_filtered_grid.shp"
anno_df = gpd.read_file(annotations_shapefile)

split_csv = "../data/grid_split_result.csv"
grid_split = pd.read_csv(split_csv)

anno_centroids = anno_df.copy()
anno_centroids["geometry"] = anno_centroids.geometry.centroid
anno_df['grid_id'] = anno_df['grid_id'].astype(int)

anno_centroids_unique = anno_centroids.drop_duplicates(subset="grid_id")[["grid_id", "geometry"]]

merged_gdf = anno_centroids_unique.merge(grid_split, on="grid_id", how="left")

merged_path = "../data/shapefile/grid/grid_split_centroids.shp"
merged_gdf.to_file(merged_path)

print(f"✅ Saved centroid+split shapefile to: {merged_path}")
