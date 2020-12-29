# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 11:07:14 2020

@author: Administrator
"""
#1->2.1->3或1->2.2->3

import numpy as np 
import matplotlib.pyplot as plt 
import cv2

###1.读取原始图像
#img = cv2.imread('F:/Anaconda/SliceImage/4/4.6.jpg')
#img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
#
#h, w = img.shape[0], img.shape[1]
#fig1 = plt.figure('原始图像')
#plt.imshow(img)
#plt.axis('off')
#plt.title('Original image')
#print( '\t\t\t   原始图像形状:\n', '\t\t\t',img.shape ) 




###2网格划分，将图像划分为m*n块

##2.1四舍五入法
#def divide_method1(img,m,n):#分割成m行n列
#    h, w = img.shape[0],img.shape[1]
#    #改global gx,gy
#    gx=np.round(gx).astype(np.int)
#    gy=np.round(gy).astype(np.int)
#
#    divide_image = np.zeros([m-1, n-1, int(h*1.0/(m-1)+0.5), int(w*1.0/(n-1)+0.5),3], np.uint8)#这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
#    for i in range(m-1):
#        for j in range(n-1):      
#            divide_image[i,j,0:gy[i+1][j+1]-gy[i][j], 0:gx[i+1][j+1]-gx[i][j],:]= img[
#                gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1],:]#这样写比a[i,j,...]=要麻烦，但是可以避免网格分块的时候，有些图像块的比其他图像块大一点或者小一点的情况引起程序出错
#    return divide_image
#
##显示分块后的图像
#def display_blocks(divide_image):#    
#    m,n=divide_image.shape[0],divide_image.shape[1]
#    for i in range(m):
#        for j in range(n):
#            plt.subplot(m,n,i*n+j+1)
#            plt.imshow(divide_image[i,j,:])
#            plt.axis('off')
#            plt.title('block:'+str(i*n+j+1))
#m=3
#n=4
#divide_image1=divide_method1(img,m+1,n+1)#该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数
#fig2 = plt.figure('分块后的子图像:四舍五入法')
#display_blocks(divide_image1)



##2.2图像缩放法:将图像缩放一下，让其满足整除关系即可。
def divide_method2(img,m,n):#分割成m行n列
    h, w = img.shape[0],img.shape[1]
    grid_h=int(h*1.0/(m-1)+0.5)#每个网格的高
    grid_w=int(w*1.0/(n-1)+0.5)#每个网格的宽
    
    #满足整除关系时的高、宽
    h=grid_h*(m-1)
    w=grid_w*(n-1)
    
    #图像缩放
    img_re=cv2.resize(img,(w,h),cv2.INTER_LINEAR)# 也可以用img_re=skimage.transform.resize(img, (h,w)).astype(np.uint8)
    #plt.imshow(img_re)
    gx, gy = np.meshgrid(np.linspace(0, w, n),np.linspace(0, h, m))
    gx=gx.astype(np.int)
    gy=gy.astype(np.int)

    divide_image = np.zeros([m-1, n-1, grid_h, grid_w,3], np.uint8)#这是一个五维的张量，前面两维表示分块后图像的位置（第m行，第n列），后面三维表示每个分块后的图像信息
    
    for i in range(m-1):
        for j in range(n-1):      
            divide_image[i,j,...]=img_re[
            gy[i][j]:gy[i+1][j+1], gx[i][j]:gx[i+1][j+1],:]
    return divide_image

#显示分块后的图像
def display_blocks(divide_image):#    
    m,n=divide_image.shape[0],divide_image.shape[1]
    for i in range(m):
        for j in range(n):
            plt.subplot(m,n,i*n+j+1)
            plt.imshow(divide_image[i,j,:])
            plt.axis('off')
            plt.title('block:'+str(i*n+j+1))
#m=3
#n=4
#divide_image2=divide_method2(img,m+1,n+1)#该函数中m+1和n+1表示网格点个数，m和n分别表示分块的块数,此为3行4列
##divide_image2=divide_method2(img,8,9)  #(img,8,9)是7行8列   此行和上三行二选一       
#fig3 = plt.figure('分块后的子图像:图像缩放法')
#display_blocks(divide_image2)




###3分块图像的还原
#分块图像的还原
#主要目的是将分块后的图像拼接起来，还原成一幅完整的大图像。有两个方法：
#1.使用opencv。其实opencv自带的有图像拼接的函数，hconcat函数：用于两个
#Mat矩阵或者图像的水平拼接；vconcat函数：用于两个Mat矩阵或者图像的垂直拼接。
#2.自己动手写。图像拼接是图像分块的逆过程，首先创建一个空的还原后的图像，
#然后将对于位置填充上对应的像素即可。由于事先不知道具体分成多少块，使用
#opencv拼接图像是很麻烦的。为了简单，我们还是选择第２种方法
def image_concat(divide_image):
    m,n,grid_h, grid_w=[divide_image.shape[0],divide_image.shape[1],#每行，每列的图像块数
                       divide_image.shape[2],divide_image.shape[3]]#每个图像块的尺寸

    restore_image = np.zeros([m*grid_h, n*grid_w, 3], np.uint8)
    restore_image[0:grid_h,0:]
    for i in range(m):
        for j in range(n):
            restore_image[i*grid_h:(i+1)*grid_h,j*grid_w:(j+1)*grid_w]=divide_image[i,j,:]
    return restore_image
##下面分别显示‘四舍五入法’和‘图像缩放法’得到的分块图像的还原结果。
#fig4 = plt.figure('分块图像的还原')
##restore_image1=image_concat(divide_image1)#四舍五入法分块还原
#restore_image2=image_concat(divide_image2)#图像缩放法分块还原
##plt.subplot(1,2,1)
##plt.imshow(restore_image1)
##plt.axis('off')
##plt.title('Rounding')
#plt.subplot(1,2,2)
#plt.imshow(restore_image2)
#plt.axis('off')
#plt.title('Scaling')
#print('\t\t\t还原后的图像尺寸')
##print('\t‘四舍五入法’：', restore_image1.shape,'\t''‘图像缩放法’：', restore_image2.shape)
#print('\t''‘图像缩放法’：', restore_image2.shape)#此行和上一行二选一
#plt.show()


