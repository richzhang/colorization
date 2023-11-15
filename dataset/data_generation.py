
import argparse
from pathlib import Path
from skimage.color import rgb2gray
from skimage.io import imread, imsave
from skimage.util import img_as_ubyte
from tqdm import tqdm

def create_dataset(
    images_path: str,
    output_path: str,
    grayscale_name_prefix: str = "gray",
    color_name_prefix: str = "color"
) -> None:
    images_path = Path(images_path)
    output_path = Path(output_path)
    grayscale_path = output_path.joinpath(grayscale_name_prefix)
    color_path = output_path.joinpath(color_name_prefix)
    try:
        grayscale_path.mkdir(parents=True, exist_ok=False)
        color_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print("Dataset folder already exists.")
        return

    length = sum(1 for _ in images_path.iterdir())

    for image_path in tqdm(images_path.iterdir(), total=length):
        image = imread(image_path.absolute().as_posix())
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
        imsave(color_image_output, image)
        # Convert from float grayscale format to uint 0-255 format
        # NOTE: this changes the values ever so slightly but is not significant
        image = img_as_ubyte(rgb2gray(image))
        imsave(grayscale_image_output, image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", help="Path to images to create dataset from")
    parser.add_argument("output_path", help="Path to output dataset to")
    parser.add_argument("-g", "--gray_prefix", default="gray", help="Name prefix for grayscale output images")
    parser.add_argument("-c", "--color_prefix", default="color", help="Name prefix for color output images")
    args = parser.parse_args()

    create_dataset(args.input_path,
                   args.output_path,
                   args.gray_prefix,
                   args.color_prefix)