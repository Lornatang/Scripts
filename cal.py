#calculate the mean and std for dataset
#The mean and std will be used in src/lib/datasets/dataset/oxfordhand.py line17-20
#The size of images in dataset must be the same, if it is not same, we can use reshape_images.py to change the size

import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
#from scipy.misc import imread
import imageio

filepath = 'GT/'  # 数据集目录
pathDir = os.listdir(filepath)
num_files = len(pathDir)

R_channel = 0
G_channel = 0
B_channel = 0
for idx in range(num_files):
    print(f"{idx:06d}/{num_files:06d}")
    filename = pathDir[idx]
    img = imageio.v2.imread(os.path.join(filepath, filename)) / 255.0
    R_channel = R_channel + np.sum(img[:, :, 0])
    G_channel = G_channel + np.sum(img[:, :, 1])
    B_channel = B_channel + np.sum(img[:, :, 2])

num = len(pathDir) * 512 * 512  # 这里（512,512）是每幅图片的大小，所有图片尺寸都一样
R_mean = R_channel / num
G_mean = G_channel / num
B_mean = B_channel / num

print("R_mean is %f, G_mean is %f, B_mean is %f" % (R_mean, G_mean, B_mean))
