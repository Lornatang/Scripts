# Copyright 2021 Dakewe Biotech Corporation. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import math
import os

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None  # Ignore read big image error problem.


def od(image: np.array, od_threshold: float, tissue_percentage: float):
    """ Check whether it's an organization or background.

    Args:
        image (np.array): Bmp read by pilot picture.
        od_threshold (float): Optical density threshold.
        tissue_percentage: Tissue percentage of own image.

    Returns:
        If tissue image return True else return False.

    """
    # Set background block numbers and tissue block numbers.
    background_num = 0
    tissue_num = 0

    # Total number of image pixels.
    pixel_num = image.size[0] * image.size[1]
    # Allocates storage for the image and loads the pixel data.
    pixel = image.load()

    for width in range(image.size[0]):
        for height in range(image.size[1]):
            od_value_r = -math.log10((pixel[width, height][0] + 1) / 256.0)
            od_value_g = -math.log10((pixel[width, height][1] + 1) / 256.0)
            od_value_b = -math.log10((pixel[width, height][2] + 1) / 256.0)
            if od_value_r > od_threshold and od_value_g > od_threshold and od_value_b > od_threshold:
                tissue_num += 1
            else:
                background_num += 1

    if (tissue_num / pixel_num) > tissue_percentage:
        return True
    else:
        return False


def main():
    od_threshold = 0.11      # Optical density threshold.
    tissue_percentage = 0.3  # Percentage of cell tissue area in the whole image.

    filenames = os.listdir("HR")
    for filename in sorted(filenames):
        print(f"Processing `{filename}`.")
        image = Image.open(os.path.join("LRunknownx4", filename))
        image = image.resize((128, 128))
        status = od(image, od_threshold, tissue_percentage)

        if not status:
            os.remove(os.path.join("HR",          filename))
            os.remove(os.path.join("LRunknownx2", filename))
            os.remove(os.path.join("LRunknownx4", filename))


if __name__ == "__main__":
    main()
