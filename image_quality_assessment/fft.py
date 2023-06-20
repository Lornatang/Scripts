import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

#################### 低通滤波反傅里叶变换 ####################
# image = cv2.imread("hr_024067.bmp", 0)
# image_float32 = np.float32(image)
# dft = cv2.dft(image_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# rows, cols = image.shape
# crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

# # 低通滤波器
# mask = np.zeros((rows, cols, 2), np.uint8)
# mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

# fshift = dft_shift * mask
# f_ishift = np.fft.ifftshift(fshift)
# image_back = cv2.idft(f_ishift)
# image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])

# plt.subplot(121), plt.imshow(image, cmap="gray")
# plt.title("Input Image"), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(image_back, cmap="gray")
# plt.title("Result"), plt.xticks([]), plt.yticks([])
# plt.show()

#################### 高通滤波反傅里叶变换 ####################
# image = cv2.imread("lr_024067.bmp", 0)
# image_float32 = np.float32(image)
# dft = cv2.dft(image_float32, flags=cv2.DFT_COMPLEX_OUTPUT)
# dft_shift = np.fft.fftshift(dft)
# rows, cols = image.shape
# crow, ccol = int(rows / 2), int(cols / 2)  # 中心位置

# # 高通滤波器
# mask = np.ones((rows, cols, 2), np.uint8)
# mask[crow-30:crow+30, ccol-30:ccol+30] = 0

# fshift = dft_shift * mask
# f_ishift = np.fft.ifftshift(fshift)
# image_back = cv2.idft(f_ishift)
# image_back = cv2.magnitude(image_back[:, :, 0], image_back[:, :, 1])

# plt.subplot(121), plt.imshow(image, cmap="gray")
# plt.title("Input Image"), plt.xticks([]), plt.yticks([])
# plt.subplot(122), plt.imshow(image_back, cmap="gray")
# plt.title("Result"), plt.xticks([]), plt.yticks([])
# plt.show()

#################### 傅里叶变换 ####################
image1 = cv2.imread("lr.bmp", 0)
f1 = np.fft.fft2(image1)
fshift1 = np.fft.fftshift(f1)
spectrum1 = 20 * np.log(np.abs(fshift1))
image2 = cv2.imread("lanczos4_lr.bmp", 0)
f2 = np.fft.fft2(image2)
fshift2 = np.fft.fftshift(f2)
spectrum2 = 20 * np.log(np.abs(fshift2))

print(mean_squared_error(image1, image2))
# plt.imshow(spectrum2, cmap="gray")
# plt.title("LR")
# plt.axis("off")
# plt.show()
