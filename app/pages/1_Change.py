from matplotlib import pyplot as plt
import streamlit as st
from glob import glob
import rasterio
import numpy as np
import matplotlib as mpl
import pandas as pd
import plotly.express as px
from modules import plot_map, plot_deforestation
from utils import multiband_to_rgb, process_image, forest_over_time

st.write("# Change in Forest Coverage")

def parse_name(item_name):
    parts = item_name.split("_")
    # return parts[5], parts[7], parts[8]
    return f"{parts[5]} - {parts[7]} - {parts[8]}"

# dropdown to select a file from data/app 
file_list = glob("data/app/S2A*")
file_names_display = [file.split("/")[-1] for file in file_list]
# file_names_display = [parse_name(file) for file in file_names_display]
file_name = st.selectbox("Select a file", file_names_display, format_func=parse_name)
# plotly
# fig = px.line(df, x=df.index, y="Forest Coverage", title="Forest Coverage Over Time")
# st.plotly_chart(fig, use_container_width=True)

# fig = px.line(df, x=df.index, y="Deforestation", title="Deforestation Over Time")
# st.plotly_chart(fig, use_container_width=True)

plot_map(file_name)
plot_deforestation(file_name)