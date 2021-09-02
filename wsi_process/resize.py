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
import cv2


def main():
    filenames = os.listdir("HR")
    for filename in sorted(filenames):
        print(f"Process `{filename}`.")
        imagex0 = cv2.imread(os.path.join("HR", filename))  # HWC.
        height, width, _ = imagex0.shape

        imagex2_nearest  = cv2.resize(imagex0, (width // 2, height // 2), interpolation=cv2.INTER_NEAREST)
        imagex2_linear   = cv2.resize(imagex0, (width // 2, height // 2), interpolation=cv2.INTER_LINEAR)
        imagex2_area     = cv2.resize(imagex0, (width // 2, height // 2), interpolation=cv2.INTER_AREA)
        imagex2_cubic    = cv2.resize(imagex0, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC)
        imagex2_lanczos4 = cv2.resize(imagex0, (width // 2, height // 2), interpolation=cv2.INTER_LANCZOS4)

        imagex4_nearest  = cv2.resize(imagex0, (width // 4, height // 4), interpolation=cv2.INTER_NEAREST)
        imagex4_linear   = cv2.resize(imagex0, (width // 4, height // 4), interpolation=cv2.INTER_LINEAR)
        imagex4_area     = cv2.resize(imagex0, (width // 4, height // 4), interpolation=cv2.INTER_AREA)
        imagex4_cubic    = cv2.resize(imagex0, (width // 4, height // 4), interpolation=cv2.INTER_CUBIC)
        imagex4_lanczos4 = cv2.resize(imagex0, (width // 4, height // 4), interpolation=cv2.INTER_LANCZOS4)
        
        cv2.imwrite(os.path.join("LRnearestx2",  filename), imagex2_nearest)
        cv2.imwrite(os.path.join("LRlinearx2",   filename), imagex2_linear)
        cv2.imwrite(os.path.join("LRareax2",     filename), imagex2_area)
        cv2.imwrite(os.path.join("LRcubicx2",    filename), imagex2_cubic)
        cv2.imwrite(os.path.join("LRlanczos4x2", filename), imagex2_lanczos4)

        cv2.imwrite(os.path.join("LRnearestx4",  filename), imagex4_nearest)
        cv2.imwrite(os.path.join("LRlinearx4",   filename), imagex4_linear)
        cv2.imwrite(os.path.join("LRareax4",     filename), imagex4_area)
        cv2.imwrite(os.path.join("LRcubicx4",    filename), imagex4_cubic)
        cv2.imwrite(os.path.join("LRlanczos4x4", filename), imagex4_lanczos4)


if __name__ == "__main__":
    main()
