import cv2
import numpy as np
from matplotlib import pyplot as plt

filename = "nobic_normal_001_244_417.bmp"

img = cv2.imread(filename)
histb = cv2.calcHist([img], [0], None, [256], [0, 255])
histg = cv2.calcHist([img], [1], None, [256], [0, 255])
histr = cv2.calcHist([img], [2], None, [256], [0, 255])

plt.plot(histb, color="b")
plt.plot(histg, color="g")
plt.plot(histr, color="r")
plt.savefig("nobic.png")

filename = "bic_normal_001_244_417.bmp"

img = cv2.imread(filename)
histb = cv2.calcHist([img], [0], None, [256], [0, 255])
histg = cv2.calcHist([img], [1], None, [256], [0, 255])
histr = cv2.calcHist([img], [2], None, [256], [0, 255])

plt.plot(histb, color="b")
plt.plot(histg, color="g")
plt.plot(histr, color="r")
plt.savefig("bic.png")