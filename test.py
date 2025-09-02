import numpy

worldclim_scale_factor = [0.1, 0.1, 1, 0.01, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 1, 1, 1, 1, 1, 1, 1, 1]
band_names = [f'bio{i:02d}' for i in range(1, 20)]

print(len(worldclim_scale_factor))
print(band_names)
