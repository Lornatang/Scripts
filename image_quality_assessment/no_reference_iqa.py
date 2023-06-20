import cv2
from glob import glob
import numpy as np
import math
from matplotlib import pyplot as plt
import time
from scipy.interpolate import make_interp_spline #平滑函数
 
 
def brenner(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰越大
    '''
    height, width = img.shape
    image_size = height * width
    out = 0.0
    
    for y in range(0, height):
        for x in range(0, width-2):
            out += (img[y, x+2])-(img[y, x])**2
        
    out /= image_size
            
    return out
 
 
def tenengrade(img):
    '''
    :param img: input image
    :return: float
    '''
    height, width = img.shape
    image_size = height * width
        
    sobelx_image = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely_image = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    sobel_image = sobelx_image*sobelx_image + sobely_image*sobely_image
    out = np.sum(sobel_image)
    out /= image_size
    
    return out
 
def Laplacian(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰值越大
    '''
    height, width = img.shape
    image_size = height * width
    out = cv2.Laplacian(img, cv2.CV_64F).var()

    out /= image_size
    
    return out
 
 
def SMD2(img):
    '''
    SMD2函数
    INPUT -> 二维灰度图像
    OUTPUT -> 图像越清晰越大
    '''
    height, width = img.shape
    image_size = height * width
    out = 0.0
    
    for y in range(0, height-1):
        for x in range(0, width-1):
            out+=math.fabs(int(img[y,x])-int(img[y,x+1]))*math.fabs(int(img[y,x]-int(img[y+1,x])))
            
    out /= image_size
    
    return out
 
 
#能量梯度
def Energy(img):
    '''
    :param img:narray 二维灰度图像
    :return: float 图像越清晰值越大
    '''
    height, width = img.shape
    image_size = height * width
    
    out = 0.0
    for y in range(0, height-1):
        for x in range(0, width-1):
            out+=((int(img[y,x+1])-int(img[y,x]))**2)+((int(img[y+1,x]-int(img[y,x])))**2)
            
    out /= image_size
    
    return out
 
 
if __name__ == '__main__':
    image_path = "LR.bmp"
    image = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray = image_gray.astype(np.float64) / 255.0
    
    output = SMD2(image_gray)
    print(f"Score: {output}")




