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

import os

import numpy as np
import openslide
from PIL import Image
from openslide.deepzoom import DeepZoomGenerator


def main():
    filenames = os.listdir("original")

    hr_sizex0  = 4096
    hr_sizex2  = 2048
    hr_sizex4  = 1024

    hr_levelx0  = 18
    hr_levelx2  = 17
    hr_levelx4  = 16

    index = 1

    for filename in sorted(filenames):
        print(f"Process `{os.path.abspath(filename)}`.")
        slide = openslide.open_slide(os.path.join("original", filename))
        slide_w, slide_h = slide.level_dimensions[0]

        num_w = int(np.floor(slide_w / hr_sizex0)) + 1
        num_h = int(np.floor(slide_h / hr_sizex0)) + 1

        data_genx0  = DeepZoomGenerator(slide, hr_sizex0,  0, False)
        data_genx2  = DeepZoomGenerator(slide, hr_sizex2,  0, False)
        data_genx4  = DeepZoomGenerator(slide, hr_sizex4,  0, False)

        for index_w in range(num_w):
            for index_h in range(num_h):
                hr_tilex0 = np.array(data_genx0.get_tile(hr_levelx0,   (index_w, index_h)))[:, :, :3]
                hr_tilex2 = np.array(data_genx2.get_tile(hr_levelx2,   (index_w, index_h)))[:, :, :3]
                hr_tilex4 = np.array(data_genx4.get_tile(hr_levelx4,   (index_w, index_h)))[:, :, :3]

                hr_imagex0 = Image.fromarray(hr_tilex0)
                hr_imagex2 = Image.fromarray(hr_tilex2)
                hr_imagex4 = Image.fromarray(hr_tilex4)

                hr_imagex0.save(os.path.join("HR",          f"{index:08d}.bmp"))
                hr_imagex2.save(os.path.join("LRunknownx2", f"{index:08d}.bmp"))
                hr_imagex4.save(os.path.join("LRunknownx4", f"{index:08d}.bmp"))

                index += 1


if __name__ == "__main__":
    main()
