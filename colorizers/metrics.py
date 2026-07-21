import numpy as np
from skimage import color
from sklearn.metrics import mean_squared_error


def get_hsv(image):
    """Calculates the mean saturation of an image in HSV color space."""
    # Convert RGB to HSV
    hsv_image = color.rgb2hsv(image)
    # Extract the saturation channel
    saturation_channel = hsv_image[:, :, 1]
    # Return the mean saturation
    return np.mean(saturation_channel)


def get_color_mse(img_orig, img_colorized):
    """Calculates the mean squared error (MSE) between the a* and b* channels of two images in LAB color space."""
    # Convert images to LAB color space
    lab_orig = color.rgb2lab(img_orig)
    lab_eccv = color.rgb2lab(img_colorized)
    
    # Extract a* and b* channels
    ab_orig = lab_orig[:, :, 1:3]
    ab_eccv = lab_eccv[:, :, 1:3]
    
    # Calculate MSE for a* and b* channels
    mse_a = mean_squared_error(ab_orig[:, :, 0], ab_eccv[:, :, 0])
    mse_b = mean_squared_error(ab_orig[:, :, 1], ab_eccv[:, :, 1])
    
    return mse_a, mse_b
