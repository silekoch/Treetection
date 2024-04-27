from os import listdir
from os.path import isfile, join

def get_file_paths(directory: str):
    return [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

def normalize_image(array, min_value, max_value):
    return ((array - min_value) / max_value)