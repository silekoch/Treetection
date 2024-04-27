import torch
from utils import get_file_paths, clip_and_normalize_image, one_hot_encode
import rasterio
import numpy as np

class ForestCoverDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data/AMAZON', mode='train', one_hot_masks=False):
        if mode not in ['train', 'val', 'test']:
            raise ValueError('Invalid mode')
        
        if mode == "train":
            self.image_dir = data_dir + '/Training/image'
            self.mask_dir = data_dir + '/Training/label'
        elif mode == "val":
            self.image_dir = data_dir + '/Validation/images'
            self.mask_dir = data_dir + '/Validation/masks'
        elif mode == "test":
            self.image_dir = data_dir + '/Training/image'
            self.mask_dir = data_dir + '/Training/mask'

        # TODO replace by scan across all files and percentile
        self.min_band_value = 0
        self.max_band_value = 2000

        self.one_hot_masks = one_hot_masks

        self.image_paths = get_file_paths(self.image_dir)
    
    def __getitem__(self, i):
        image_path = self.image_paths[i]
        with rasterio.open(image_path) as src:
            # Read the raster bands
            red_band = src.read(1)
            green_band = src.read(2)
            blue_band = src.read(3)

            # Combine the bands into one rgb image
            rgb_array = np.array([red_band, green_band, blue_band]).transpose(1, 2, 0)
            image = clip_and_normalize_image(
                rgb_array, self.min_band_value, self.max_band_value)

        mask_path = self.mask_dir + '/' + image_path.split('/')[-1]
        with rasterio.open(mask_path) as src:
            mask = np.array(src.read(1), dtype=np.float32)

        if self.one_hot_masks:
            mask = one_hot_encode(mask)
        else:
            mask = np.expand_dims(mask, axis=0)

        return image, mask
        
    def __len__(self):
        return len(self.image_paths)
