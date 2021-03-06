# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 18:22:10 2020

@author: Administrator
"""
 
import matplotlib.pyplot as plt 
import cv2 as cv
from block import divide_method2
from block import display_blocks
import os
import numpy as np
import cv2
import sift02
##1.读取原始图像imgA、imgB
#imgA = cv2.imread("F:/Anaconda/SliceImage/4/4.5.jpg", cv2.COLOR_BGR2GRAY)#为./img/l/left.jpg
#imgB = cv2.imread("F:/Anaconda/SliceImage/4/4.6.jpg", cv2.COLOR_BGR2GRAY)#为./img/r/right.jpg

img01 = cv2.imread(r"F:/Anaconda/SliceImage/4/4.5.jpg")
img01 = cv2.cvtColor(img01,cv2.COLOR_BGR2RGB)
imgA = img01[120:660,100:640] #获取感兴趣区域
 

hA, wA = imgA.shape[0], imgA.shape[1]
fig1 = plt.figure('原始图像A')
plt.imshow(imgA)
plt.axis('off')
plt.title('Original image A')
print( '\t\t\t   原始图像A形状:\n', '\t\t\t',imgA.shape ) 

m=3
n=4
divide_image2A=divide_method2(imgA,m+1,n+1)#该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
fig3 = plt.figure('分块后的子图像A:图像缩放法')
display_blocks(divide_image2A)


img02 = cv2.imread(r'F:/Anaconda/SliceImage/4/4.6.jpg')
img02 = cv2.cvtColor(img02,cv2.COLOR_BGR2RGB)
imgB = img02[120:660,100:640]

hB, wB = imgB.shape[0], imgB.shape[1]
fig1 = plt.figure('原始图像B')
plt.imshow(imgB)
plt.axis('off')
plt.title('Original image  B')
print( '\t\t\t   原始图像B形状:\n', '\t\t\t',imgB.shape ) 

m=3
n=4
divide_image2B=divide_method2(imgB,m+1,n+1)#该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
fig3 = plt.figure('分块后的子图像B:图像缩放法')
display_blocks(divide_image2B)

if hA==hB and wA==wB:
        print ('A、B大小相同，可以分块匹配')
        #sift = cv.xfeatures2d.SIFT_create()
        m,n=divide_image2A.shape[0],divide_image2A.shape[1]
        for i in range(m):
            for j in range(n):
                img1=divide_image2A[i,j,:]
#                img2=divide_image2B[i,j,:]
                img2=imgB
#                sift = cv.xfeatures2d.SURF_create()
#                kp1,des1 = sift.detectAndCompute(img1,None)
#                kp2,des2 = sift.detectAndCompute(img2,None)
#                my_match2(img1,
#                img2,kp1,kp2,des1,des2,
#                cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
                sift02(img1,img2)
    

else:
        print ('A、B大小不同，分块配准失败。')