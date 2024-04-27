from os import listdir
from os.path import isfile, join
import numpy as np

def get_file_paths(directory: str):
    return [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

def clip_and_normalize_image(array, min_value, max_value):
    array = np.clip(array, min_value, max_value)
    return ((array - min_value) / max_value)