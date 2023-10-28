
import argparse
import cv2
from pathlib import Path
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
        image = cv2.imread(image_path.absolute().as_posix())
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
        cv2.imwrite(color_image_output, image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        L, _, _ = cv2.split(image)
        cv2.imwrite(grayscale_image_output, L)

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