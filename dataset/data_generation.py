
import argparse
import numpy as np
from pathlib import Path
from skimage.color import rgb2gray, rgb2lab
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from tqdm import tqdm

from dataset.cielab import CIELabConversion
from dataset.data_utils import resize_image

import typing as T

def create_dataset(
    images_path: str,
    output_path: str,
    buckets_path: str = "resources/buckets_313.npy",
    grayscale_name_prefix: str = "gray",
    color_name_prefix: str = "color",
    bucket_label_prefix: str = "bucket",
    resize_image_size: T.Union[T.Tuple[int, int], None] = (256, 256)
) -> None:
    cielab = CIELabConversion(buckets_path=buckets_path)

    images_path = Path(images_path)
    output_path = Path(output_path)
    grayscale_path = output_path.joinpath(grayscale_name_prefix)
    color_path = output_path.joinpath(color_name_prefix)
    bucket_labels_path = output_path.joinpath(bucket_label_prefix)

    try:
        grayscale_path.mkdir(parents=True, exist_ok=False)
        color_path.mkdir(parents=True, exist_ok=False)
        bucket_labels_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Dataset folder already exists.")
        return

    length = sum(1 for _ in images_path.iterdir())

    for image_path in tqdm(images_path.iterdir(), total=length):
        image = imread(image_path.absolute().as_posix())
        if resize_image_size:
            image = resize_image(image, shape=resize_image_size)

        color_image_output = (
            color_path.joinpath(color_name_prefix +
                                "_" +
                                image_path.stem).absolute().as_posix() +
            ".png"
        )
        grayscale_image_output = (
            grayscale_path.joinpath(grayscale_name_prefix +
                                    "_" +
                                    image_path.stem).absolute().as_posix() +
            ".png"
        )
        bucket_label_output = (
            bucket_labels_path.joinpath(bucket_label_prefix +
                                    "_" +
                                    image_path.stem).absolute().as_posix() +
            ".npy"
        )
        imsave(color_image_output, img_as_ubyte(image))

        np.save(bucket_label_output, cielab.get_image_ab_buckets(rgb2lab(image)))

        # Convert from float grayscale format to uint 0-255 format
        # NOTE: this changes the values ever so slightly but is not significant
        image = img_as_ubyte(rgb2gray(image))
        imsave(grayscale_image_output, image)


if __name__ == "__main__":
    from ast import literal_eval
    def none_or_tuple(value):
        if value == 'None':
            return None
        return literal_eval(value)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to images to create dataset from")
    parser.add_argument("output_path", help="Path to output dataset to")
    parser.add_argument("-g", "--gray_prefix", default="gray", help="Name prefix for grayscale output images")
    parser.add_argument("-c", "--color_prefix", default="color", help="Name prefix for color output images")
    parser.add_argument("-b", "--bucket_prefix", default="bucket", help="Name prefix for image bucket labels")
    parser.add_argument("--buckets_path", default="resources/buckets_313.npy", help="Path to quantized ab-value buckets")
    parser.add_argument("-r", "--resize-image-size", default="(256, 256)", type=none_or_tuple, help="(w, h) tuple to resize images to or None")
    args = parser.parse_args()

    create_dataset(args.input_path,
                   args.output_path,
                   args.buckets_path,
                   args.gray_prefix,
                   args.color_prefix,
                   args.bucket_prefix,
                   args.resize_image_size)
