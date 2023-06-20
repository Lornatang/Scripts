import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--sr", type=str)
parser.add_argument("--hr", type=str)
args = parser.parse_args()

src = cv2.imread(args.sr, 0)
dst = cv2.imread(args.hr, 0)

# 0. 检查图像尺寸是否相同.
assert src.shape == dst.shape, "Image size not equal."
assert src.shape[0] == src.shape[1], "Image width and height is not equal."
N = src.shape[0]

# 1. 横向计算图像灰度直方图.
all_hist_src = []
all_hist_dst = []
for hist_height in range(N):
    # 计算每行灰度直方图.
    hist_src = cv2.calcHist([src[hist_height, :]], [0], None, [N], [0, 255])
    hist_dst = cv2.calcHist([dst[hist_height, :]], [0], None, [N], [0, 255])
    all_hist_src.append(hist_src)
    all_hist_dst.append(hist_dst)

# 2. 1D傅里叶变换(截取单边数据).
all_spectrum_src = []
all_spectrum_dst = []
for index in range(N):
    # 快速傅里叶变换
    fft_src = np.fft.fft(all_hist_src[index])
    fft_dst = np.fft.fft(all_hist_dst[index])
    # 取复数的绝对值，即复数的模(双边频谱).
    spectrum_src = np.abs(fft_src)
    spectrum_dst = np.abs(fft_dst)
    # 由于对称性，只取一半区间(单边频谱).
    spectrum_src = spectrum_src[range(N//2)]
    spectrum_dst = spectrum_dst[range(N//2)]
    all_spectrum_src.append(spectrum_src)
    all_spectrum_dst.append(spectrum_dst)

# 3. 求光谱平均.
avg_spectrum_src = []
avg_spectrum_dst = []
# 遍历N个光谱图中0~(N//2)范围内的光谱数值.
for spectrum in range(N//2):
    total_spectrum_src = 0
    total_spectrum_dst = 0
    for index in range(N):
        total_spectrum_src += all_spectrum_src[index][spectrum]
        total_spectrum_dst += all_spectrum_dst[index][spectrum]
    avg_spectrum_src.append(total_spectrum_src / N)
    avg_spectrum_dst.append(total_spectrum_dst / N)

# plt.figure()
# plt.title("Image")
# plt.xlabel("Bins")
# plt.ylabel("Spectrum")
# plt.xlim([0, N//2])
# plt.ylim([-0.1, 2.5])
# plt.plot(avg_spectrum_src, color="green", label=args.sr)
# plt.plot(avg_spectrum_dst, color="red",   label=args.hr)
# plt.legend()
# plt.show()


# 4. 利用公式求差异.
def calc_diff(spectrum_src, spectrum_dst):
    diff = 0.
    # N = len(spectrum_dst)
    for index in range(N//2):
        diff += (spectrum_dst[index] - spectrum_src[index]) ** 2
    return np.sqrt(diff / (N/2))


print(f"`{args.sr}` and `{args.hr}` diff is: {float(calc_diff(avg_spectrum_src, avg_spectrum_dst)):.8f}.")