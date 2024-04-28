import torch
from utils import get_file_paths, clip_and_normalize_image, one_hot_encode
import rasterio
import numpy as np

class ForestCoverDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir='data/AMAZON', mode='train', one_hot_masks=False, overfitting_mode=None, NIR=False):
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
        self.NIR = NIR

        if overfitting_mode == 'sample':
            self.image_paths = get_file_paths(self.image_dir)[:1]
        if overfitting_mode == '2sample':
            self.image_paths = get_file_paths(self.image_dir)[:2]
        elif overfitting_mode == 'batch':
            self.image_paths = get_file_paths(self.image_dir)[:16]
        else:
            self.image_paths = get_file_paths(self.image_dir)
    
    def __getitem__(self, i):
        image_path = self.image_paths[i]
        with rasterio.open(image_path) as src:
            # Read the rgb bands
            bands = [src.read(1), src.read(2), src.read(3)]
            if self.NIR:
                bands.append(src.read(4))

            # Combine the bands into one image
            image_array = np.array(bands, dtype=np.float32)
            image = clip_and_normalize_image(
                image_array, self.min_band_value, self.max_band_value)

        mask_path = self.mask_dir + '/' + image_path.split('/')[-1]
        with rasterio.open(mask_path) as src:
            mask = np.array(src.read(1), dtype=np.float32)

        if self.one_hot_masks:
            mask = one_hot_encode(mask).transpose(2, 0, 1)
        else:
            mask = np.expand_dims(mask, axis=0)

        return image, mask

    def get_item_HWC(self, i):
        image, mask = self.__getitem__(i)
        return image.transpose(1, 2, 0), mask

    def __len__(self):
        return len(self.image_paths)
