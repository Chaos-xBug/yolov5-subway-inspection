import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import json
from markline import predict as mp
import time

def Compute_Sift(image01, image02):
    MIN_MATCH_COUNT = 5
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image01, None)
    kp2, des2 = sift.detectAndCompute(image02, None)
    ### 后续优化，使用surf特征提高速度，通过保存读取模板特征节省计算

    # kdtree建立索引方式的常量参数
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # checks指定索引树要被遍历的次数
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    img11 = cv2.drawKeypoints(image01, kp1, None)
    plt.imshow(img11), plt.show()
    img22 = cv2.drawKeypoints(image02, kp2, None)
    plt.imshow(img22), plt.show()
    # drawPrams = dict(matchColor=(0, 255, 0),
    #                  singlePointColor=(255, 0, 0),
    #                  matchesMask=good,
    #                  flags=0)
    # img3 = cv2.drawMatchesKnn(image01, kp1, image02, kp2, matches, None, **drawPrams)
    # img_PutText = cv2.putText(img3, "SIFT+kNNMatch: Image Similarity Comparisonn", (40, 40), cv2.FONT_HERSHEY_COMPLEX,
    #                           1.5,
    #                           (0, 0, 255), 3, )
    # img4 = cv2.resize(img_PutText, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)  # 缩小1/2
    # plt.imshow(img4), plt.title('matches'), plt.show()

    # cv2.imshow("matches", img4)
    # cv2.waitKey(70000)
    # cv2.destroyAllWindows()
    return kp1, kp2, good

def Compute_ORB(image01, image02):
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(image01, None)
    kp2, des2 = orb.detectAndCompute(image02, None)
    ### 后续优化，使用surf特征提高速度，通过保存读取模板特征节省计算

    # kdtree建立索引方式的常量参数
    flann = cv2.BFMatcher()
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.5* n.distance:
            good.append(m)
    # print(len(good))
    img11 = cv2.drawKeypoints(image01, kp1, None)
    cv2.imwrite(save_path+'\img11.jpg', img11)
    img22 = cv2.drawKeypoints(image02, kp2, None)
    # cv2.imwrite(save_path + 'manques\\img22.bmp', img22)
    plt.imshow(img22), plt.show()
    return kp1, kp2, good

def Transform_Sansac(image1, image2, kp1, kp2, good):  # 模板图像 待检图像

    MIN_MATCH_COUNT = 5
    if len(good) > MIN_MATCH_COUNT:
        # 改变数组的表现形式，不改变数据内容，数据内容是每个关键点的坐标位置
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        # findHomography 函数是计算变换矩阵
        # 参数cv2.RANSAC是使用RANSAC算法寻找一个最佳单应性矩阵H，即返回值M
        # 返回值：M 为变换矩阵，mask是掩模
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        h, w, l = image1.shape
        # pts是图像img1的四个顶点
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)  # 计算变换后的四个顶点坐标位置


        Minv = cv2.getPerspectiveTransform(dst, pts)  # 逆变换矩阵
        warped = cv2.warpPerspective(image2, Minv, (w, h), flags=cv2.INTER_LINEAR)  # 逆变图像
        coordinates = cv2.perspectiveTransform(pts, Minv)  # image2变换后的顶点坐标
        # print('pts.shape:', pts.shape)
        # print('pts:', pts)
        # print('coordinates.shape:', coordinates.shape)
        # print('coordinates:', coordinates)

        return warped, Minv
    else:

        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return -1, 0

def template_match(image1, image2):

    kp1, kp2, matches = Compute_Sift(image1, image2)
    warped, Minv = Transform_Sansac(image1, image2, kp1, kp2, matches) # 得到image2的变换图像
    # try:
    #     if warped == -1:
    #         continue
    # except:
    #     print("can't match' + filename)
    # plt.imshow(img_check3), plt.title('Test Image'), plt.show()
    # plt.imshow(img_o3), plt.title('Standard Image'), plt.show()
    # plt.imshow(warped), plt.title('Warped Standard Image'), plt.show()
    return warped, Minv

if __name__ == "__main__":
    '''
    初始化定义
    '''
    count = 0
    path_base = 'data\\Line3\\'
    path_label_o = 'data\\Line3\\StandardDefautTree\\labels\\'
    path_label_check = 'data\\Line3\\DefautTestResult\\labels\\'

    filename = '17.bmp'
    count = count + 1
    path1 = os.path.join(path_base, "Standard\\", filename) # 模板图片
    path2 = os.path.join(path_base, "Defaut\\", filename) # 原始待检测图片
    save_path = os.path.join(path_base, "manques\\", filename) # 缺失项点截取图像
    '''
    Step1 模板匹配
    '''
    img_o3 = cv2.imread(path1, 1)
    img_check3 = cv2.imread(path2, 1)

    warped, Minv = template_match(img_check3, img_o3)

    plt.imshow(img_check3), plt.title('Test Image'), plt.show()
    # plt.imshow(img_o3), plt.title('Standard Image'), plt.show()
    plt.imshow(warped), plt.title('Warped Standard Image'), plt.show()