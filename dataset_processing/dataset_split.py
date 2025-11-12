import geopandas as gpd
import pandas as pd
import numpy as np
import pulp
from pathlib import Path
import shutil
from tqdm import tqdm


REMOVE_SPECIES = [49, 62, 64]
dataset_split_ratio = {'train': 0.7, 'validation':0.15, 'test':0.15}
TOL_PERCENT = 0.02
alpha = 0.7
eps = 1e-9

annotations_shp = "../data/shapefile/grid/swiss_tree_annotations_with_filtered_grid.shp"
anno_df = gpd.read_file(annotations_shp)
anno_df['grid_id'] = anno_df['grid_id'].astype(int)

tree_species_list = anno_df['TreeAI_ID'].unique()

tree_species_list = tree_species_list[~np.isin(tree_species_list, REMOVE_SPECIES)]
anno_df = anno_df[anno_df['TreeAI_ID'].isin(tree_species_list)]

tree_distribution = pd.crosstab(index=anno_df['grid_id'], columns=anno_df['TreeAI_ID'])
tree_distribution = tree_distribution.loc[:, tree_distribution.sum(axis=0) > 0]

splits = list(dataset_split_ratio.keys())
grids   = tree_distribution.index.tolist()
species = tree_distribution.columns.tolist()

T   = tree_distribution.sum(axis=0)
T_k = pd.DataFrame({k: dataset_split_ratio[k]*T for k in splits}).T

G = len(grids)
target_n = {k: int(round(dataset_split_ratio[k] * G)) for k in splits}
tol = max(1, int(TOL_PERCENT*G))
lower = {k: max(0, target_n[k] - tol) for k in splits}
upper = {k: min(G, target_n[k] + tol) for k in splits}

# Begin ILP search
m = pulp.LpProblem("grid_split", pulp.LpMinimize)

x = pulp.LpVariable.dicts("x", (grids, splits), lowBound=0, upBound=1, cat="Binary")

S = pulp.LpVariable.dicts("S", (splits, species), lowBound=0)

dpos = pulp.LpVariable.dicts("dpos", (splits, species), lowBound=0)
dneg = pulp.LpVariable.dicts("dneg", (splits, species), lowBound=0)

for g in grids:
    m += pulp.lpSum(x[g][k] for k in splits) == 1, f"one_split_per_grid[{g}]"

for k in splits:
    for s in species:
        m += S[k][s] == pulp.lpSum(x[g][k] * float(tree_distribution.at[g, s]) for g in grids), f"S_def[{k},{s}]"

for k in splits:
    for s in species:
        m += S[k][s] - float(T_k.loc[k, s]) == dpos[k][s] - dneg[k][s], f"abs_lin[{k},{s}]"

for k in splits:
    m += pulp.lpSum(x[g][k] for g in grids) >= lower[k], f"lower_count[{k}]"
    m += pulp.lpSum(x[g][k] for g in grids) <= upper[k], f"upper_count[{k}]"

w = {s: 1.0 / (float(T[s]) + eps)**alpha for s in species}
cap = 10 * np.median(list(w.values()))
w = {s: min(ws, cap) for s, ws in w.items()}
mean_w = np.mean(list(w.values()))
w = {s: ws / (mean_w + 1e-12) for s, ws in w.items()}
m += pulp.lpSum((dpos[k][s] + dneg[k][s]) * w[s] for k in splits for s in species), "weighted_L1_deviation"

solver = pulp.PULP_CBC_CMD(
    msg=True,
    threads=6,
    options=[
        "ratioGap=1e-5",
    ],
)
res = m.solve(solver)

assign = pd.Series({g: max(splits, key=lambda k: pulp.value(x[g][k])) for g in grids}, name="split")

achieved = tree_distribution.join(assign).groupby("split").sum().reindex(splits).fillna(0)
deviation_abs = (achieved - T_k.values).abs()

print("Status:", pulp.LpStatus[res])
print("Total |abs deviation|:", deviation_abs.to_numpy().sum())
print("Count per split (grids):")
print(assign.value_counts())

out_csv = "../data/grid_split_result.csv"
assign_df = pd.DataFrame({
    "grid_id": assign.index.astype(int),
    "split": assign.values
})
assign_df.to_csv(out_csv, index=False)


# Regroup the folder to different splits.
# grid_split = pd.read_csv('../data/grid_split_result.csv')
# hdf5_filepath = Path('../data/TreeAI_Swiss_60/HDF5')
# hdf5_files = hdf5_filepath.rglob('*.hdf5')
#
# for hdf5_file in tqdm(hdf5_files, desc='Splitting the dataset'):
#     grid_id = int(hdf5_file.stem)
#     split = grid_split.loc[grid_split['grid_id'] == grid_id, 'split'].iloc[0]
#     dst_path = Path('../data/TreeAI_Swiss_60/splits/') / split / hdf5_file.name
#     dst_path.parent.mkdir(exist_ok=True, parents=True)
#     shutil.copy(src=hdf5_file, dst=dst_path)

