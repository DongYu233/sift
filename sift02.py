# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 17:41:45 2020

@author: Administrator
"""

#coding=utf-8
'''
@project : binocular_vision
@author  : Hoodie_Willi
#@description: $输出sift匹配后，匹配点的坐标
#@time   : 2019-05-28 10:25:36
'''

#sift匹配，出结果，求出特征点个数，匹配上的特征点的序号及该点在图1和图2中的坐标，匹配点为蓝点, 坏点为红点。
import numpy as np
import cv2
#img01 = cv2.imread("F:/Anaconda/SliceImage/4/4.5.jpg", cv2.COLOR_BGR2GRAY)#为./img/l/left.jpg,左图
#img02 = cv2.imread("F:/Anaconda/SliceImage/4/4.6.jpg", cv2.COLOR_BGR2GRAY)#为./img/r/right.jpg开口
#img1 = img01[120:660,100:640] #获取感兴趣区域
#img2 = img02[120:660,100:640] 
##[数字变大缩图片上面：数字变小缩图片下面，数字变大缩左边：数字变小缩右边]，蝗虫切片图像[120:640,100:620]
##cv2.namedWindow('match',cv2.WINDOW_NORMAL)#
##cv2.imshow('match',img)#
##cv2.imwrite("sift.png",img)
##cv2.waitKey(2000000)
##cv2.destroyAllWindows()
##cv2.waitKey(1)

def sift02(img1,img2):
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)#sift.detectAndComputer(gray， None)  
    kp2, des2 = sift.detectAndCompute(img2, None)# 计算出图像的关键点和sift特征向量参数说明：gray表示输入的图片
    bf =cv2.BFMatcher(normType=cv2.NORM_L2,crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)#matcher.knnMatch(featuresA, featuresB, 2)
                                        #featuresA和featuresB是两幅图片的特征向量，该函数的返回值是一个DMatch，DMatch是一个匹配之后的集合。
                                        #DMatch中的每个元素含有三个参数：
                                        #queryIdx：测试图像的特征点描述符的下标（第几个特征点描述符），同时也是描述符对应特征点的下标。
                                        #trainIdx：样本图像的特征点描述符下标,同时也是描述符对应特征点的下标。
                                        #distance：代表这怡翠匹配的特征点描述符的欧式距离，数值越小也就说明俩个特征点越相近。
     
    # ## Create flann matcher
    # FLANN_INDEX_KDTREE = 1 # bug: flann enums are missing
    # flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    # #matcher = cv2.FlannBasedMatcher_create()
    # matcher = cv2.FlannBasedMatcher(flann_params, {})
     
    ## Ratio test
    print(len(matches))
    
    matchesMask = [[0, 0] for i in range(len(matches))]#len(matches)个[0, 0]
    #{原：匹配成功的点的序号
    #for i, (m1, m2) in enumerate(matches):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    #                                    #enumerate(sequence, [start=0]) #sequence：一个序列、迭代器或其他支持迭代对象。start：】【下标起始位置。返回 enumerate(枚举) 对象。
    #    if m1.distance < 0.7 * m2.distance:# 两个特征向量之间的欧氏距离，越小表明匹配度越高。
    #        matchesMask[i] = [1, 0]
    #        pt1 = kp1[m1.queryIdx].pt  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
    #        pt2 = kp2[m1.trainIdx].pt  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号
    #        # print(kpts1)
    #        print(i, pt1, pt2)#匹配成功的点的序号，图1中该点的坐标，图2中该点的序号
    #原}
    List1 = []
    List2 = []
    
    #{加：匹配成功的点的序号，计算匹配的点的斜率，
    for i, (m1, m2) in enumerate(matches):#enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
                                        #enumerate(sequence, [start=0]) #sequence：一个序列、迭代器或其他支持迭代对象。start：】【下标起始位置。返回 enumerate(枚举) 对象。
        if m1.distance < 0.74 * m2.distance:# 两个特征向量之间的欧氏距离，越小表明匹配度越高。
            matchesMask[i] = [1, 0]  
            pt1 = np.float32(kp1[m1.queryIdx].pt)  # trainIdx    是匹配之后所对应关键点的序号，第一个载入图片的匹配关键点序号
            pt2 = np.float32(kp2[m1.trainIdx].pt)  # queryIdx  是匹配之后所对应关键点的序号，第二个载入图片的匹配关键点序号     
                #print("pt1[0]= %f,pt1[1]= %f,img1.shape[0]=%f,img1.shape[1]=%f" % (pt1[0] , pt1[1],img1.shape[0],img1.shape[1]))
                #128 [238.2185  433.81155] [254.27953 456.3411 ]
                #pt1[0]= 238.218506,pt1[1]= 433.811554,img1.shape[0]=644.000000,img1.shape[1]=800.000000
            k=(int(pt2[1])-int(pt1[1]))/((img1.shape[1])+int(pt2[0])-(int(pt1[0])))#求斜率
            #匹配成功的点的序号，图1中该点的坐标，图2中该点的序号     
            print(i,k,'(',int(pt1[0]),',',int(pt1[1]),')','(',img1.shape[1]+int(pt2[0]),',',int(pt2[1]),')') #img1(644,800,3),img2(644,800,3),res(644,1600,3)
            #print(i,k,'(',int(pt1[0]),',',int(pt1[1]),')','(',int(pt2[0]),',',int(pt2[1]),')')
            #List = []
    #        List.append(k)
            if k>=0:
                List1.append(k)#依次存入计算出来的斜率
            else:
                List2.append(k)
            
        
    #加}
            if i % 5 ==0:
                cv2.circle(img1, (int(pt1[0]),int(pt1[1])), 5, (255,0,255), -1)
                cv2.circle(img2, (int(pt2[0]),int(pt2[1])), 5, (255,0,255), -1)
                #cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
                #img：输入的图片data
                #center：圆心位置
                #radius：圆的半径
                #color：圆的颜色
                #thickness：圆形轮廓的粗细（如果为正）。负厚度表示要绘制实心圆。
                #lineType： 圆边界的类型。
                #shift：中心坐标和半径值中的小数位数。
    
    #{这三个值，对找与众不同的斜率没有帮助
    if len(List1)>0:
        #求均值
        arr_mean = np.mean(List1)
        #求方差
        arr_var = np.var(List1)
        #求标准差
        arr_std = np.std(List1,ddof=1)
        print("+平均值为：%f" % arr_mean)
        print("+方差为：%f" % arr_var)
        print("+标准差为:%f" % arr_std)
    if len(List2)>0:
        arr_mean = np.mean(List2)
        #求方差
        arr_var = np.var(List2)
        #求标准差
        arr_std = np.std(List2,ddof=1)
        print("-平均值为：%f" % arr_mean)
        print("-方差为：%f" % arr_var)
        print("-标准差为:%f" % arr_std)
    #}
    
    
    
    
    
    
    
    
        
    # 匹配点为蓝点, 坏点为红点
    draw_params = dict(matchColor = (255, 0,0),
            singlePointColor = (0,0,255),
            matchesMask = matchesMask,
            flags = 0)
     
    res = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None,**draw_params)
    
    #cv2.imshow("Result", res)
    #
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    cv2.namedWindow('match',cv2.WINDOW_NORMAL)#
    cv2.imshow('match',res)#
    cv2.imwrite("sift.png",res)
    cv2.waitKey(2000000)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    



 

