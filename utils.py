from os import listdir
from os.path import isfile, join
import numpy as np

def get_file_paths(directory: str):
    return [join(directory, f) for f in listdir(directory) if isfile(join(directory, f))]

def clip_and_normalize_image(array, min_value, max_value):
    array = np.clip(array, min_value, max_value)
    return ((array - min_value) / max_value)

# Perform one hot encoding on label
def one_hot_encode(mask):
    one_hot_mask = np.zeros((mask.shape[0], mask.shape[1], 2))
    one_hot_mask[mask == 0, 0] = 1
    one_hot_mask[mask == 1, 1] = 1
    assert(np.sum(one_hot_mask) == mask.shape[0] * mask.shape[1])
    return one_hot_mask
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(one_hot_mask):
    return np.argmax(one_hot_mask, axis=-1)