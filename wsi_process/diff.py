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
import sys


def reportdiff(unique1, unique2, dir1, dir2):
    """
    生成目录差异化报告
    """
    if not (unique1 or unique2):
        print("Directory lists are identical")
    else:
        if unique1:
            print('Files unique to：', dir1)
            for file in unique1:
                print('....', file)
        if unique2:
            print('Files unique to：', dir2)
            for file in unique2:
                print('.........', file)


def difference(seq1, seq2):
    """
    仅返回seq1中的所有项
    """
    return [item for item in seq1 if item not in seq2]


def comparedirs(dir1, dir2, files1=None, files2=None):
    """
    比较文件的名字
    """
    print('Comparing...', dir1, 'to....', dir2)
    files1 = os.listdir(dir1) if files1 is None else files1
    files2 = os.listdir(dir2) if files2 is None else files2
    unique1 = difference(files1, files2)
    unique2 = difference(files2, files1)
    reportdiff(unique1, unique2, dir1, dir2)
    return not (unique1, unique2)


if __name__ == "__main__":
    dir1, dir2 = sys.argv[1:]
    comparedirs(dir1, dir2)
