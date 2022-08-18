


import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import math


def detect_line(img):
    # 进行中值滤波
    dst_img = cv2.blur(img, [3, 3])
    # plt.imshow(dst_img), plt.title('DST_Image'), plt.show()

    hsv_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)
    # plt.imshow(hsv_img), plt.title('HSV_Image'), plt.show()

    mask1 = cv2.inRange(hsv_img, np.array([0, 63, 33]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_img, np.array([163, 80, 50]), np.array([180, 255, 255]))
    mask = mask1 | mask2
    # plt.imshow(mask, 'gray'), plt.title('Mask'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    open =cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel2)
    # plt.imshow(open), plt.title('Open'), plt.show()
    return close


basepath = r'E:\04-Data\02-robot\04-Xingqiao0512\desmonstration\Two'

for filename in os.listdir(basepath):
    # filename = basepath + 'KeyholeCap41_Pic_2022_05_13_150620_3.bmp'
    # filename = basepath + 'KeyholeCap40_Pic_2022_05_13_150722_7.bmp'
    print(filename)
    img = cv2.imread(os.path.join(basepath,filename))

    res = detect_line(img)

    contoursred, hierarchy = cv2.findContours(res, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(contoursred)
    print(hierarchy)
    # 根据红线区域点拟合直线
    line = cv2.fitLine(contoursred[1], cv2.DIST_L2, 0, 0.01, 0.01)
    angle = math.degrees(math.asin(line[1]))
    # 画出拟合红线的直线
    k = line[1] / line[0]
    b = line[3] - k * line[2]
    img_s = img.copy()
    res_s = np.zeros((res.shape[0],res.shape[1],3))
    for i in range(3):
        res_s[:,:,i] = res.copy()

    # cv2.line(img_s, (int(line[2]),int(line[3])), (0, int(b)), (0, 0, 255))
    # cv2.line(res_s, (int(line[2]),int(line[3])), (0, int(b)), (0, 0, 255))

    cv2.line(img_s, (200,int(200*k)+int(b)), (0, int(b)), (0, 0, 255))
    cv2.line(res_s, (200,int(200*k)+int(b)), (0, int(b)), (0, 0, 255))

    # if len(contoursred) >= 2:
    # 根据红线区域点拟合直线
    line2 = cv2.fitLine(contoursred[0], cv2.DIST_L2, 0, 0.01, 0.01)
    angle2 = math.degrees(math.asin(line2[1]))
    # 画出拟合红线的直线
    k2 = line2[1] / line2[0]
    b2 = line2[3] - k2 * line2[2]

    cv2.line(img_s, (200,int(200*k2)+int(b2)), (0, int(b2)), (0, 0, 255))
    cv2.line(res_s, (200,int(200*k2)+int(b2)), (0, int(b2)), (0, 0, 255))

    # cv2.line(img_s, (int(line2[2]),int(line2[3])), (0, int(b2)), (0, 0, 255))
    # cv2.line(res_s, (int(line2[2]),int(line2[3])), (0, int(b2)), (0, 0, 255))
    # plt.subplot(221)
    # plt.imshow(img[:, :, ::-1]),plt.axis('off')                #显示彩色图片  参数默认     去掉坐标系
    # plt.subplot(222)
    # plt.imshow(res,cmap=plt.cm.gray),plt.axis('off')       #显示彩色图片 参数设为灰度图片
    # plt.subplot(223)
    # plt.imshow(img_s),plt.axis('off')
    # plt.imshow(res)                     #显示灰度图片  参数默认
    # plt.subplot(224)
    # plt.imshow(res_s,cmap=plt.cm.gray),plt.axis('off')    #显示灰度图片 参数设为灰度图片
    # plt.show()

    cv2.namedWindow('1', 0)
    cv2.resizeWindow('1', 150, 150)
    cv2.moveWindow('1', 600, 80)
    cv2.imshow('1', img)

    cv2.namedWindow('2', 0)
    cv2.resizeWindow('2', 150, 150)
    cv2.moveWindow('2', 770, 80)
    cv2.imshow('2', res)

    cv2.namedWindow('3', 0)
    cv2.resizeWindow('3', 150, 150)
    cv2.moveWindow('3', 600, 280)
    cv2.imshow('3', res_s)

    cv2.namedWindow('4', 0)
    cv2.resizeWindow('4', 150, 150)
    cv2.moveWindow('4', 770, 280)
    cv2.imshow('4', img_s)
    cv2.waitKey(2500)
