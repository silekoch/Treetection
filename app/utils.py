import numpy as np
import pandas as pd
import rasterio
from glob import glob

def clip_and_normalize_image(array, min_value, max_value):
    array = np.clip(array, min_value, max_value)
    return ((array - min_value) / max_value)

def multiband_to_rgb(image):
    rgb = image[:3].transpose(1, 2, 0)
    rgb = clip_and_normalize_image(rgb, 0, 2000)
    return rgb

def process_image(image, squeeze=False):
    # normalize image to cut off all values above 99th percentile
    # then move to 0-255
    if squeeze:
        bound = np.percentile(image, 95)
        image[image > bound] = bound
    image = (image / image.max() * 255).astype(np.uint8)
    return image

def total_deforestation(name, promise=None):
    label_names = sorted(glob(f"data/app/{name}/*/label.tif"))
    if not label_names:
        return 0
    label_start = promise/100 if promise else rasterio.open(label_names[0]).read().mean()
    label_end = rasterio.open(label_names[-1]).read().mean()
    # return (label_start - label_end) / label_start * 100
    return (1 - label_end / label_start) * 100

def forest_over_time(file_name):
    date_list = glob(f"data/app/{file_name}/*")
    date_names = [date.split("/")[-1] for date in date_list]
    date_names.sort()
    labels = []
    for date in sorted(date_list):
        labels.append(rasterio.open(f'{date}/label.tif').read().mean()*100)
    df = pd.DataFrame(labels, index=date_names, columns=["Forest Coverage"])
    df["Deforestation"] = -df["Forest Coverage"].diff()
    return df
