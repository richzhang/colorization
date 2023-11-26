
import os
import typing as T

import numpy as np
from skimage.util import img_as_ubyte
from skimage.color import rgb2lab
from skimage.io import imread
from torch.utils.data import Dataset

from .cielab import CIELabConversion
from .data_utils import resize_image

class ColorizationDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        cielab_conversion: CIELabConversion = CIELabConversion(),
        grayscale_name_prefix: str = "gray",
        color_name_prefix: str = "color",
        bucket_label_prefix: str = "bucket",
        resize_image_size: T.Union[T.Tuple[int, int], None] = (256, 256)
    ) -> None:
        self.grayscale_name_prefix = grayscale_name_prefix
        self.color_name_prefix = color_name_prefix
        self.bucket_label_prefix = bucket_label_prefix

        self.grayscale_path = os.path.join(dataset_path, grayscale_name_prefix)
        self.color_path = os.path.join(dataset_path, color_name_prefix)
        self.bucket_path = os.path.join(dataset_path, bucket_label_prefix)

        # NOTE: This may need to change if we can't load all filenames in memory
        self.grayscale_images = os.listdir(self.grayscale_path)
        self.color_images = os.listdir(self.color_path)
        self.bucket_labels = os.listdir(self.bucket_path)

        self.cielab = cielab_conversion
        self.resize_image_size = resize_image_size

    def __len__(self) -> int:
        return len(self.grayscale_images)

    def __getitem__(self, index: int) -> T.Tuple[np.ndarray, np.ndarray]:
        grayscale_image_path = self.grayscale_images[index]
        color_image_path = (self.color_name_prefix +
                            "_" +
                            grayscale_image_path.removeprefix(self.grayscale_name_prefix + "_"))
        # There's probably a better way to do this
        bucket_label_path = (self.bucket_label_prefix +
                             "_" +
                             grayscale_image_path.removeprefix(self.grayscale_name_prefix + "_"))[:-4] + ".npy"
        
        grayscale_image = imread(os.path.join(self.grayscale_path, grayscale_image_path), as_gray=True)
        if self.resize_image_size:
            grayscale_image = resize_image(grayscale_image, shape=self.resize_image_size)
            grayscale_image = img_as_ubyte(grayscale_image) # resize automatically converts to float 0-1 grayscale format
        
        color_image = imread(os.path.join(self.color_path, color_image_path))
        if self.resize_image_size:
            color_image = resize_image(color_image, shape=self.resize_image_size)
        color_image = rgb2lab(color_image)

        bucket_ids = np.load(os.path.join(self.bucket_path, bucket_label_path))

        # TODO: Preprocessing

        return (grayscale_image, bucket_ids)