from glob import glob
import matplotlib as mpl
import rasterio
import streamlit as st

from utils import multiband_to_rgb, process_image, forest_over_time


def plot_map(file_name):
    # slider for dates
    date_list = glob(f"data/app/{file_name}/*")
    date_names = [date.split("/")[-1] for date in date_list]
    print(file_name)
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
        st.image(multiband_to_rgb(image), caption="Band 0", clamp=True)
    with col_img2:
        cmap = mpl.colormaps['YlGn']
        st.image(cmap(process_image(label[0], squeeze=True)), caption="Forest Coverage")
        
def plot_deforestation(file_name):
    df = forest_over_time(file_name)
    st.line_chart(df, height=300, y="Forest Coverage")
    st.line_chart(df, height=300, y="Deforestation")

