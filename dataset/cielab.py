import numpy as np

import typing as T

# Hardcoded min/max ranges for the buckets
BUCKET_A_MIN = -90
BUCKET_A_MAX = 100
BUCKET_B_MIN = -110
BUCKET_B_MAX = 100

class CIELabConversion():
    def __init__(
        self,
        buckets_path: str,
    ) -> None:
        """
        Instantiates class for converting CIELab ab-values to Buckets.

        Args:
            buckets_path (str): Path to .npy file containing all the quantized ab-value
                buckets.
            buckets_knn_path (str): Path to .joblib file containing the Nearest-Neighbor
                classifier for finding the closest bucket to an ab-value.

        Attributes:
            buckets (np.ndarray): NumPy array containing all the quantized ab-value
                buckets.
            ab2bucket (dict[tuple[int, int], int]): Dict for converting a quantized ab-value
                to corresponding bucket index.
            bucket2ab (dict[int, tuple[int, int]]): Dict for converting a bucket index to
                corresponding ab-value.
        """
        self.buckets = np.load(buckets_path)
        self.ab2bucket = {tuple(self.buckets[i]): i+1 for i in range(len(self.buckets))}
        self.bucket2ab = {i+1: tuple(self.buckets[i]) for i in range(len(self.buckets))}
        
    def get_image_ab_buckets(self, image_lab: np.ndarray) -> np.ndarray:
        """
        Get bucketed ab-values from Lab image.
        """
        # Extract only ab-values
        image_ab = image_lab[:, :, 1:]
        orig_ab_shape = image_ab.shape

        # Reshape to essentially 1D with 2 element ab-values
        image_ab = image_ab.reshape(-1, 2)

        # Round to nearest 10 (abusing our bucket values)
        image_ab = np.round(image_ab, -1)
        # Clamp values to valid mapping values
        image_ab[:, 0] = np.clip(image_ab[:, 0], BUCKET_A_MIN, BUCKET_A_MAX)
        image_ab[:, 1] = np.clip(image_ab[:, 1], BUCKET_B_MIN, BUCKET_B_MAX)
        image_ab = image_ab.astype(int)
        # Apply ab to bucket mapping
        for i in range(len(image_ab)):
            image_ab[i] = self.ab2bucket[tuple(image_ab[i])] # changes [a, b] to bucket [x, x]
        image_ab = image_ab[:, 0] # remove duplicate bucket value

        # Reshape to original shape
        image_ab = image_ab.reshape((orig_ab_shape[0], orig_ab_shape[1]))

        return image_ab

    def convert_buckets_to_ab(self, image_buckets: np.ndarray) -> np.ndarray:
        """
        Get ab-values from bucketed image.
        """
        converted_image_ab = []
        for row in image_buckets:
            new_row = []
            for i in row:
                new_row.append(self.bucket2ab[i])
            converted_image_ab.append(new_row)

        converted_image_ab = np.array(converted_image_ab)

        return converted_image_ab