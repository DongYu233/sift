# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 14:47:06 2020

@author: Administrator
"""

#二维图像的离散变余弦换（DCT）
#Python3.5
#库：cv2+numpy+matplotlib
#作者：James_Ray_Murphy
import os 
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
 
img0 = cv2.imread('F:/Anaconda/SliceImage/4/4.5.jpg', 0)

img = img0[120:640,100:620] #获取感兴趣区域
#[数字变大缩图片上面：数字变小缩图片下面，数字变大缩左边：数字变小缩右边]，蝗虫切片图像[120:640,100:620]
#cv2.imshow('match',img)

#cv2.namedWindow('match',cv2.WINDOW_NORMAL)#
#cv2.imshow('match',img)#
#cv2.imwrite("sift.png",img)
#cv2.waitKey(2000000)
#cv2.destroyAllWindows()
#cv2.waitKey(1)

#坐标
#plt.subplot(111)
#plt.imshow(img0, 'gray')
#plt.title('original image')





 
#img1 = img.astype('float')
# 
#C_temp = np.zeros(img.shape)
#dst = np.zeros(img.shape)
# 
# 
#m, n = img.shape
#N = n
#C_temp[0, :] = 1 * np.sqrt(1/N)
# 
#for i in range(1, m):
#     for j in range(n):
#          C_temp[i, j] = np.cos(np.pi * i * (2*j+1) / (2 * N )
#) * np.sqrt(2 / N )
# 
#dst = np.dot(C_temp , img1)
#dst = np.dot(dst, np.transpose(C_temp))
# 
#dst1= np.log(abs(dst))  #进行log处理
# 
#img_recor = np.dot(np.transpose(C_temp) , dst)
#img_recor1 = np.dot(img_recor, C_temp)
# 
##自带方法
# 
#img_dct = cv2.dct(img1)         #进行离散余弦变换
# 
#img_dct_log = np.log(abs(img_dct))  #进行log处理
# 
#img_recor2 = cv2.idct(img_dct)    #进行离散余弦反变换
#
#
# 
# 
#plt.subplot(231)
#plt.imshow(img1, 'gray')
#plt.title('original image')
#plt.xticks([]), plt.yticks([])
#
# 
#plt.subplot(232)
#plt.imshow(dst1)
#plt.title('DCT1')
#plt.xticks([]), plt.yticks([])
# 
#
# 
#plt.subplot(233)
#plt.imshow(img_recor1, 'gray')
#plt.title('IDCT1')
#plt.xticks([]), plt.yticks([])
#
# 
#plt.subplot(234)
#plt.imshow(img, 'gray')
#plt.title('original image')
# 
#plt.subplot(235)
#plt.imshow(img_dct_log)
#plt.title('DCT2(cv2_dct)')
#
#
#
# 
#plt.subplot(236)
#plt.imshow(img_recor2,'gray')
#plt.title('IDCT2(cv2_idct)')
#
# 
#plt.show()