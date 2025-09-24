1. The pipeline of **Sentinel-1 data preprocessing** comes from:

   Mullissa, A.; Vollrath, A.; Odongo-Braun, C.; Slagter, B.; Balling, J.; Gou, Y.; Gorelick, N.; Reiche, J. Sentinel-1 SAR Backscatter Analysis Ready Data Preparation in Google Earth Engine. Remote Sens. 2021, 13, 1954. https://doi.org/10.3390/rs13101954

2. For monthly composite Sentinel-1 image, it seems that mean() is better than medium(). We need to find some papers to support this idea.s

3. The  vegetation height model comes from:
   Ginzler, C. (2021). Vegetation Height Model NFI.  National Forest Inventory (NFI).  https://www.doi.org/10.16904/1000001.1.

4. For climate data, we use CHELSA v2.1 dataset to replace WorldClim, because the former is more precise in mountainous area. Website: https://chelsa-climate.org/

5. 