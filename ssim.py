import numpy as np
import cv2
import heapq
import os
from PIL import Image
from scipy.signal import convolve2d
from matplotlib import pyplot as plt


def getListMaxNumIndex(num_list,topk=3):
    '''
    获取列表中最大的前n个数值的位置索引
    '''
    print(type(num_list))
    print(num_list.shape)
    num = num_list.tolist()
    max_num_index = map(num.index, heapq.nlargest(topk, num))
    return list(max_num_index)
    # min_num_index=map(num_list.index, heapq.nsmallest(topk,num_list))

def getMatrixMaxNumIndex(num_list, n=3):
    arr = num_list.ravel()
    flat_indices = np.argpartition(arr, (len(arr)-n))[-n:]   # 通过argpartition切分，输出前n个最大值
    row_indices, col_indices = np.unravel_index(flat_indices,num_list.shape)
    return row_indices, col_indices

def getMatrixMinNumIndex(num_list, n=3):
    arr = num_list.ravel()
    flat_indices = np.argpartition(arr, n-1)[:n]   # 通过argpartition切分，输出前n个最小值
    row_indices, col_indices = np.unravel_index(flat_indices,num_list.shape)
    return row_indices, col_indices

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)

def CutEdges(im1, warped, length):
    h,w = im1.shape
    im1 = im1[length:h-2*length,length:w-2*length]
    warped = warped[length:h - 2 * length, length:w - 2 * length]
    return im1, warped

def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=21, L=255):

    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)  # 算子
    window = window / np.sum(np.sum(window))  # 归一化

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    # ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    # 去掉亮度对比
    ssim_map = (2 * sigmal2 + C2) / (sigma1_sq + sigma2_sq + C2)
    # 计算ssim值
    ssim_mean = np.mean(np.mean(ssim_map))

    return ssim_mean, ssim_map

def Compute_Sift(image01, image02):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(image01, None)
    kp2, des2 = sift.detectAndCompute(image02, None)

    # # feature match
    # orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(image01, None)
    # kp2, des2 = orb.detectAndCompute(image02, None)

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
        matchesMask = mask.ravel().tolist()  # ravel()展平，并转成列表

        h, w = image1.shape
        # pts是图像img1的四个顶点
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        # 根据四个顶点坐标位置在img1图像画出的边框
        # image1 = cv2.polylines(image1, [np.int32(pts)], True, 0, 1, cv2.LINE_AA)
        # plt.imshow(image1, 'gray'), plt.show()
        # print(pts.shape)
        dst = cv2.perspectiveTransform(pts, M)  # 计算变换后的四个顶点坐标位置
        # print(dst[0,1,2])

        # 根据四个顶点坐标位置在img2图像画出变换后的边框
        # image2 = cv2.polylines(image2, [np.int32(dst)], True, 255, 1, cv2.LINE_AA)
        # plt.imshow(image2, 'gray'), plt.show()

        # img_size = (int(dst[2, 0, 0] - dst[0, 0, 0]), int(dst[1, 0, 1] - dst[0, 0, 1]))
        # print(img_size)
        Minv = cv2.getPerspectiveTransform(dst, pts)
        warped = cv2.warpPerspective(image2, Minv, (w, h), flags=cv2.INTER_LINEAR)
        return warped
    else:

        print("Not enough matches are found - %d/%d" % (len(good), MIN_MATCH_COUNT))
        matchesMask = None
        return -1


    # # 画出特征点
    # img1 = cv2.drawKeypoints(image1, kp1, None)
    # plt.imshow(img1), plt.show()
    # img2 = cv2.drawKeypoints(image2, kp2, None)
    # plt.imshow(img2), plt.show()
    # 画出对应关系
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv2.drawMatches(image1, kp1, image2, kp2, good, None, **draw_params)
    plt.imshow(img3, 'gray'), plt.show()

def seg_kmeans_gray(img):
    # 展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 1))
    img_flat = np.float32(img_flat)

    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 20, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 进行聚类
    compactness, labels, centers = cv2.kmeans(img_flat, 2, None, criteria, 10, flags)

    # 显示结果
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    return img_output


if __name__ == "__main__":

    # im1 = cv2.imread(r'E:\04-Data\cutimg\SplitPin\4.jpg', 0)  # 模板
    # im2 = cv2.imread(r'E:\04-Data\cutimg\SplitPin\1.jpg', 0)  # 待检测图像

    # im1 = cv2.imread(r'E:\03-PHM\testpic\test02.png', 0)
    # im2 = cv2.imread(r'E:\03-PHM\testpic\test01.png', 0)

    # im1 = cv2.imread(r'E:\03-PHM\testpic\6.jpg', 0)
    # im2 = cv2.imread(r'E:\03-PHM\testpic\5.jpg', 0)

    im11 = cv2.imread(r'E:\03-Anomaly Detection\try\T\naviID_4_devID_1_pos_1_1.jpg', 0)
    im22 = cv2.imread(r'E:\03-Anomaly Detection\try\N\naviID_4_devID_1_pos_1_1.jpg', 0)

    plt.imshow(im11, 'gray'), plt.show()
    plt.imshow(im22, 'gray'), plt.show()

    im1 = cv2.resize(im11, (224, 224))
    im2 = cv2.resize(im22, (224, 224))
    # m,n = im1.shape
    # print(m)
    # print(n)
    # fx = 224/n
    # fy = 224/m
    # im1 = cv2.resize(im1, None, fx = fx, fy = fy)
    # im2 = cv2.resize(im2, None, fx = fx, fy = fy)


    kp1, kp2, matches = Compute_Sift(im1, im2)
    warped = Transform_Sansac(im1, im2, kp1, kp2, matches)
    length = 40
    im1, warped = CutEdges(im1, warped, length)


    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Source_image')
    ax1.imshow(im1, 'gray')
    ax2.set_title('warped_image')
    ax2.imshow(warped, 'gray')
    plt.show()

    plt.imshow(im1, 'gray'), plt.show()
    plt.imshow(warped, 'gray'), plt.show()


    # # 平滑滤波
    # kernel = 13
    # im1 = cv2.blur(im1, (kernel, kernel))
    # warped = cv2.blur(warped, (kernel, kernel))

    # cv2.imwrite(r'C:\Users\Lenovo\PycharmProjects\Change-detection-master\13.png', im1)
    # cv2.imwrite(r'C:\Users\Lenovo\PycharmProjects\Change-detection-master\14.png', warped)


    '''
    ssim特征
    '''
    # 比对
    ssim_value, ssim_map = compute_ssim(np.array(im1), np.array(warped), win_size = 5)
    plt.imshow(ssim_map, 'gray'), plt.show()

    index = np.unravel_index(ssim_map.argmax(), ssim_map.shape)
    print(ssim_value)

    row, col = getMatrixMinNumIndex(ssim_map, 1300)
    plt.imshow(ssim_map, 'gray')
    plt.plot(col, row, 'o')
    plt.show()


'''
聚类分析
'''

    # # 聚类
    # im1_k = seg_kmeans_gray(im1)
    # warped_k = seg_kmeans_gray(warped)
    # # 画聚类效果对比
    # plt.subplot(121), plt.imshow(im1, 'gray'), plt.title('input')
    # plt.subplot(122), plt.imshow(im1_k, 'gray'), plt.title('kmeans')
    # plt.show()
    # # 画两种图片聚类结果对比
    # print(im1_k)
    # warped_k = 1-warped_k
    # plt.subplot(121), plt.imshow(im1_k, 'gray'), plt.title('kmeans_input')
    # plt.subplot(122), plt.imshow(warped_k, 'gray'), plt.title('kmeans_warped')
    # plt.show()
    #
    #
    # # 差分
    # # img_x = ((im1_k-warped_k) <= 0) * 255
    # img_x = (im1_k==warped_k) * 255
    # plt.imshow(img_x, 'gray'), plt.show()


