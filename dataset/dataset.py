
import os
import typing as T

import numpy as np
from skimage.color import rgb2lab
from skimage.io import imread
from torch.utils.data import Dataset

from .cielab import CIELabConversion

class ColorizationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        cielab_conversion: CIELabConversion = CIELabConversion(),
        grayscale_name_prefix: str = "gray",
        color_name_prefix: str = "color"
    ) -> None:
        self.grayscale_name_prefix = grayscale_name_prefix
        self.color_name_prefix = color_name_prefix

        self.grayscale_path = os.path.join(dataset_path, grayscale_name_prefix)
        self.color_path = os.path.join(dataset_path, color_name_prefix)
        # NOTE: This may need to change if we can't load all filenames in memory
        self.grayscale_images = os.listdir(self.grayscale_path)
        self.color_images = os.listdir(self.color_path)

        self.cielab = cielab_conversion

    def __len__(self) -> int:
        return len(self.grayscale_images)

    def __getitem__(self, index: int) -> T.Tuple[np.ndarray, np.ndarray]:
        grayscale_image_path = self.grayscale_images[index]
        color_image_path = (self.color_name_prefix +
                            "_" +
                            grayscale_image_path.removeprefix(self.grayscale_name_prefix + "_"))
        
        grayscale_image = imread(os.path.join(self.grayscale_path, grayscale_image_path), as_gray=True)
        
        color_image = imread(os.path.join(self.color_path, color_image_path))
        color_image = rgb2lab(color_image)

        bucket_ids = self.cielab.get_image_ab_buckets(color_image)

        # TODO: Preprocessing

        return (grayscale_image, bucket_ids)