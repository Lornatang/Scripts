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


def main():
    index = 1
    filenames = os.listdir("HR")
    for filename in sorted(filenames):
        print(f"Process `{filename}`.")
        os.rename(os.path.join("HR",           filename), os.path.join("HR",           f"{index:08d}.bmp"))
        os.rename(os.path.join("LRnearestx2",  filename), os.path.join("LRnearestx2",  f"{index:08d}.bmp"))
        os.rename(os.path.join("LRnearestx4",  filename), os.path.join("LRnearestx4",  f"{index:08d}.bmp"))
        os.rename(os.path.join("LRlinearx2",   filename), os.path.join("LRlinearx2",   f"{index:08d}.bmp"))
        os.rename(os.path.join("LRlinearx4",   filename), os.path.join("LRlinearx4",   f"{index:08d}.bmp"))
        os.rename(os.path.join("LRareax2",     filename), os.path.join("LRareax2",     f"{index:08d}.bmp"))
        os.rename(os.path.join("LRareax4",     filename), os.path.join("LRareax4",     f"{index:08d}.bmp"))
        os.rename(os.path.join("LRcubicx2",    filename), os.path.join("LRcubicx2",    f"{index:08d}.bmp"))
        os.rename(os.path.join("LRcubicx4",    filename), os.path.join("LRcubicx4",    f"{index:08d}.bmp"))
        os.rename(os.path.join("LRlanczos4x2", filename), os.path.join("LRlanczos4x2", f"{index:08d}.bmp"))
        os.rename(os.path.join("LRlanczos4x4", filename), os.path.join("LRlanczos4x4", f"{index:08d}.bmp"))
        os.rename(os.path.join("LRunknownx2",  filename), os.path.join("LRunknownx2",  f"{index:08d}.bmp"))
        os.rename(os.path.join("LRunknownx4",  filename), os.path.join("LRunknownx4",  f"{index:08d}.bmp"))
        index += 1


if __name__ == "__main__":
    main()
