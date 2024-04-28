import streamlit as st
import leafmap.foliumap as leafmap

st.set_page_config(layout="wide")

st.title("Split-panel Map with GeoTIFF")

# Initialize the map
m = leafmap.Map()

# Paths to your GeoTIFF files
geotiff_path_left = "data/app/S2B_MSIL2A_20200202T141649_N0213_R010_T20LQN_20200202T164215_18_10/2018-01-12/label.tif"
geotiff_path_right = "data/app/S2B_MSIL2A_20200202T141649_N0213_R010_T20LQN_20200202T164215_18_10/2024-04-26/label.tif"
# geotiff_path_left = "data/AMAZON/Training/label/S2B_MSIL2A_20200202T141649_N0213_R010_T20LQN_20200202T164215_18_10.tif"
# geotiff_path_left = "data/app/.tif"
geotiff_path_right = "data/AMAZON/Training/label/S2B_MSIL2A_20200202T141649_N0213_R010_T20LQN_20200202T164215_18_10.tif"
# geotiff_path_right = geotiff_path_left

geotiff_path_left = "data/AMAZON/Training/label/S2A_MSIL2A_20200111T142701_N0213_R053_T20NQG_20200111T164651_01_07.tif"
geotiff_path_right = "data/AMAZON/Training/label/S2A_MSIL2A_20200111T142701_N0213_R053_T20NQG_20200111T164651_01_13.tif"

import rasterio as rio

# Open the GeoTIFFs
with rio.open(geotiff_path_left) as src:
    print(src.profile)
    print(src.bounds)
    print(src.crs)
with rio.open("data/app/S2B_MSIL2A_20200202T141649_N0213_R010_T20LQN_20200202T164215_18_10/2018-01-12/label.tif") as src:
    print(src.profile)
    print(src.bounds)
    print(src.crs)

# Adding GeoTIFF layers to the map
m.add_raster(
    geotiff_path_left,
    layer_name="Left Layer",
    palette="viridis",
    opacity=0.7
)

m.add_raster(
    geotiff_path_right,
    layer_name="Right Layer",
    palette="viridis",
    opacity=0.7
)

# Use split_map to display both GeoTIFFs side by side
m.split_map(left_layer="Left Layer", right_layer="Right Layer")

# Optionally add a legend (customize as needed)
m.add_legend(title='Sample Legend', legend_dict={'Low': 'green', 'Medium': 'yellow', 'High': 'red'})

# Display the map in Streamlit
m.to_streamlit(height=700)


# import rasterio
# from rasterio.merge import merge
# from rasterio.plot import show
# import glob
# import os


# tif_files = glob.glob("data/AMAZON/Training/label/*.tif")[:5]

# # List of open rasterio datasets
# src_files_to_mosaic = [rasterio.open(fp) for fp in tif_files]

# # Merge function
# mosaic, out_trans = merge(src_files_to_mosaic)

# # Output path for the merged result
# out_fp = "data/app/merged.tif"

# # Copy the metadata
# out_meta = src_files_to_mosaic[0].meta.copy()

# # Update the metadata
# out_meta.update({
#     "driver": "GTiff",
#     "height": mosaic.shape[1],
#     "width": mosaic.shape[2],
#     "transform": out_trans,
#     "crs": src_files_to_mosaic[0].crs
# })

# # Write the mosaic raster to disk
# with rasterio.open(out_fp, "w", **out_meta) as dest:
#     dest.write(mosaic)