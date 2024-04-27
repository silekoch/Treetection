from matplotlib import pyplot as plt
import streamlit as st
from glob import glob
import rasterio
import numpy as np
import matplotlib as mpl
import pandas as pd
import plotly.express as px

def process_image(image, squeeze=False):
    # normalize image to cut off all values above 99th percentile
    # then move to 0-255
    if squeeze:
        bound = np.percentile(image, 95)
        image[image > bound] = bound
    image = (image / image.max() * 255).astype(np.uint8)
    return image

st.write("# Change in Forest Coverage")

# dropdown to select a file from data/app 
file_list = glob("data/app/*")
file_names_display = [file.split("/")[-1] for file in file_list]
file_name = st.selectbox("Select a file", file_names_display)

# slider for dates
date_list = glob(f"data/app/{file_name}/*")
date_names = [date.split("/")[-1] for date in date_list]
date_names.sort()

date1 = st.select_slider("Select date", date_names, value=date_names[-1])


image_dir = f"data/app/{file_name}/{date1}"
image_path = f"{image_dir}/image.tif"
image = rasterio.open(image_path).read()
label_path = f"{image_dir}/label.tif"
label = rasterio.open(label_path).read()
mask_path = f"{image_dir}/mask.tif"
mask = rasterio.open(mask_path).read()

col_img1, col_img2 = st.columns(2)
with col_img1:
    st.image(process_image(image[0], squeeze=True), caption="Band 0")
with col_img2:
    cmap = mpl.colormaps['YlGn']
    st.image(cmap(process_image(label[0], squeeze=True)), caption="Forest Coverage")

labels = []
for date in sorted(date_list):
    # st.write(rasterio.open(f'{date}/label.tif').read()[0].mean())
    labels.append(rasterio.open(f'{date}/label.tif').read().mean()*100)
df = pd.DataFrame(labels, index=date_names, columns=["Forest Coverage"])
df["Deforestation"] = -df["Forest Coverage"].diff()
# st.line_chart(df, height=300)
# plotly
fig = px.line(df, x=df.index, y="Forest Coverage", title="Forest Coverage Over Time")
st.plotly_chart(fig, use_container_width=True)

fig = px.line(df, x=df.index, y="Deforestation", title="Deforestation Over Time")
st.plotly_chart(fig, use_container_width=True)
