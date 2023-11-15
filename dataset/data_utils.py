import numpy as np
from skimage.transform import resize
from skimage.util import img_as_float

def resize_image(
    image: np.ndarray,
    shape: tuple[int, int] = (256, 256)
) -> np.ndarray:
    return resize(image, shape, anti_aliasing=True)


def merge_grayscale_image_ab_to_lab(grayscale: np.ndarray, image_ab: np.ndarray) -> np.ndarray:
    # grayscale in uint form
    image_lab = np.zeros((grayscale.shape[0], grayscale.shape[1], 3))
    # Convert from uint form to float form, then to luminance 0-100 range
    image_lab[:, :, 0] = img_as_float(grayscale) * 100
    image_lab[:, :, 1:] = image_ab

    return image_lab