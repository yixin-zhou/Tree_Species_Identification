import numpy
import os

worldclim_scale_factor = [0.1, 0.1, 1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1]
band_names = [f'bio{i:02d}' for i in range(1, 20)]

print(len(worldclim_scale_factor))
print(band_names)

core_count = os.cpu_count()
print(f"This computer has {core_count} CPU cores.")