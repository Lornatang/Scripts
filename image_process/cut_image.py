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
"""The image cutting method used in the conventional SR data set adds the offset parameter."""
import os

from PIL import Image


def main(filename, image_size, stride):
    """

    Args:
        filename (str): Image file name, usually a relative path or an absolute path.
        image_size (int): split image size, (height = width).
        stride (int): How much is the difference between each cutting area and the next cutting area.
    """
    print(f"Process `{os.path.basename(filename)}`.")

    image = Image.open(filename)
    width, height = image.size

    # Initialize the cutting area position.
    left, upper, right, lower = 0, 0, image_size, image_size

    index = 1
    while right <= width:
        while lower <= height:
            # crop: (left, upper, right, lower).
            image_crop = image.crop((left, upper, right, lower))
            image_crop.save(filename.split(".")[-2] + f"_{str(index)}." + filename.split(".")[-1])
            upper = upper + stride
            lower = upper + image_size
            index += 1
        left = left + stride
        right = left + image_size
        upper = 0
        lower = image_size

    os.remove(filename)


if __name__ == "__main__":
    dirname = "original"
    image_size = 32
    stride = 14

    files = os.listdir(dirname)
    for file in files:
        main(os.path.join(dirname, file), image_size, stride)
