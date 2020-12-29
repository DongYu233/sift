# -*- coding: utf-8 -*-
"""
Created on Tue Aug 18 23:10:27 2020

@author: chy
"""

import numpy as np
from sklearn.decomposition import PCA
import time 
import cv2 as cv
import cv2
import matplotlib.pyplot as plt
def my_match2(img1,img2,kp1,kp2,des1,des2,mc_type):
    time1= time.time() 
    bf = cv.BFMatcher(normType=cv2.NORM_L2,crossCheck=False)
    matches = bf.knnMatch(des1,des2,k=2) 
    #此处的2就是每个匹配装两个对应的点位
    good = []
    better = [] 
    max1,max2 = 0,0
    #不难发现，m,n就是对应刚才的2
    for m,n in matches:
        if m.distance < 0.75*n.distance:#参数越低，准确率越高，但是匹配出来的点越少。
            good.append([m])  
            better.append(m)                
#    print(len(good))
    print("匹配到"+str(len(good))+"个点")#此行和上行二选一

        
            
    img3 = cv.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=mc_type)
    #cv.imshow('match',img3)
    cv.namedWindow('match, 0.75*n.distance',cv2.WINDOW_NORMAL)#
    cv.imshow('match, 0.75*n.distance',img3)#
    cv2.imwrite("sift.png",img3)
#    cv.waitKey(0)
    cv.waitKey(50000)#此行和上行二选一
    cv.destroyAllWindows()
    cv.waitKey(1)

#    src_pts = np.array([kp1[i.queryIdx].pt for i in better])#.reshape(-1, 1, 2)
#    dst_pts = np.array([kp2[i.trainIdx].pt for i in better])#.reshape(-1, 1, 2)
#    ransacReprojThreshold = 1.0#过滤掉一部分配错的点
#    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold)
#    mask = mask.ravel()
#    kp11 = src_pts[mask==1]
#    kp22 = dst_pts[mask==1]
#    for i in range(len(kp11)):
#        if max1<kp11[i][0]-kp22[i][0]:
#            max1=kp11[i][0]-kp22[i][0]
       

#img1 = cv.imread(r'F:/Anaconda/SliceImage/4/4.5.jpg')
#
#img2 = cv.imread(r'F:/Anaconda/SliceImage/4/4.6.jpg')
#
#
#
#
##sift = cv.xfeatures2d.SIFT_create()
#sift = cv.xfeatures2d.SURF_create()
#kp1,des1 = sift.detectAndCompute(img1,None)
#kp2,des2 = sift.detectAndCompute(img2,None)
#
#
#my_match2(img1,
#img2,kp1,kp2,des1,des2,
#cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    
