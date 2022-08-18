#msssim.py
import numpy as np
import scipy.signal
import cv2
from matplotlib import pyplot as plt

def Transform_Sansac(image1, image2, kp1, kp2, good):  # 模板图像 待检图像
    MIN_MATCH_COUNT = 6
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
        return image2


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
    search_params = dict(checks=25)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.95 * n.distance:
            good.append(m)
    return kp1, kp2, good

def Compute_Surf(image01, image02):
    # Initiate SIFT detector
    surf = cv2.xfeatures2d.SURF_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = surf.detect(image01, None)
    kp2, des2 = surf.detect(image02, None)

    # # feature match
    # orb = cv2.ORB_create()
    # kp1, des1 = orb.detectAndCompute(image01, None)
    # kp2, des2 = orb.detectAndCompute(image02, None)

    # kdtree建立索引方式的常量参数
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # checks指定索引树要被遍历的次数
    search_params = dict(checks=25)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good.append(m)
    return kp1, kp2, good

def Compute_ORB(image01, image02):
    # Initiate SIFT detector
    orb = cv2.ORB_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = orb.detectAndCompute(image01, None)
    kp2, des2 = orb.detectAndCompute(image02, None)
    ### 后续优化，使用surf特征提高速度，通过保存读取模板特征节省计算

    # kdtree建立索引方式的常量参数
    BF = cv2.BFMatcher()
    matches = BF.knnMatch(des1, des2, k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.95* n.distance:
            good.append(m)
    # print(len(good))
    return kp1, kp2, good

def ssim_index_new(img1,img2,K,win):

    M,N = img1.shape

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (K[0]*255)**2
    C2 = (K[1]*255) ** 2
    win = win/np.sum(win)

    mu1 = scipy.signal.convolve2d(img1,win,mode='valid')
    mu2 = scipy.signal.convolve2d(img2,win,mode='valid')
    mu1_sq = np.multiply(mu1,mu1)
    mu2_sq = np.multiply(mu2,mu2)
    mu1_mu2 = np.multiply(mu1,mu2)
    sigma1_sq = scipy.signal.convolve2d(np.multiply(img1,img1),win,mode='valid') - mu1_sq
    sigma2_sq = scipy.signal.convolve2d(np.multiply(img2, img2), win, mode='valid') - mu2_sq
    img12 = np.multiply(img1, img2)
    sigma12 = scipy.signal.convolve2d(np.multiply(img1, img2), win, mode='valid') - mu1_mu2

    if(C1 > 0 and C2>0):
        ssim1 =2*sigma12 + C2
        ssim_map = np.divide(np.multiply((2*mu1_mu2 + C1),(2*sigma12 + C2)),np.multiply((mu1_sq+mu2_sq+C1),(sigma1_sq+sigma2_sq+C2)))
        cs_map = np.divide((2*sigma12 + C2),(sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2*mu1_mu2 + C1
        numerator2 = 2*sigma12 + C2
        denominator1 = mu1_sq + mu2_sq +C1
        denominator2 = sigma1_sq + sigma2_sq +C2

        ssim_map = np.ones(mu1.shape)
        index = np.multiply(denominator1,denominator2)
        #如果index是真，就赋值，是假就原值
        n,m = mu1.shape
        for i in range(n):
            for j in range(m):
                if(index[i][j] > 0):
                    ssim_map[i][j] = numerator1[i][j]*numerator2[i][j]/denominator1[i][j]*denominator2[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]
        for i in range(n):
            for j in range(m):
                if((denominator1[i][j] != 0)and(denominator2[i][j] == 0)):
                    ssim_map[i][j] = numerator1[i][j]/denominator1[i][j]
                else:
                    ssim_map[i][j] = ssim_map[i][j]

        cs_map = np.ones(mu1.shape)
        for i in range(n):
            for j in range(m):
                if(denominator2[i][j] > 0):
                    cs_map[i][j] = numerator2[i][j]/denominator2[i][j]
                else:
                    cs_map[i][j] = cs_map[i][j]


    mssim = np.mean(ssim_map)
    # plt.imshow(ssim_map), plt.show()
    # plt.imshow(cs_map), plt.show()
    mcs = np.mean(cs_map)

    return  mssim,mcs, ssim_map

def CutEdges(im1, warped, length):
    h,w = im1.shape
    s, p = warped.shape
    im1 = im1[length:h-length,length:w-length]
    warped = warped[length:h - length, length:w - length]
    return im1, warped

def msssim(img1,img2):

    K = [0.01,0.03]
    win  = np.multiply(cv2.getGaussianKernel(11, 1.5), (cv2.getGaussianKernel(11, 1.5)).T)  # H.shape == (r, c)
    level = 5
    weight = [0.2448,0.2856,0.2001,0.1363,0.1333]
    method = 'product'

    M,N = img1.shape
    S,P = img2.shape
    H,W = win.shape

    downsample_filter = np.ones((2,2))/4
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mssim_array = []
    mcs_array = []

    for i in range(0,level):
        mssim,mcs,ssim_map = ssim_index_new(img1,img2,K,win)
        # plt.imshow(ssim_map,'gray'), plt.title('result_map'), plt.show()
        mssim_array.append(mssim)
        mcs_array.append(mcs)
        filtered_im1 = cv2.filter2D(img1,-1,downsample_filter,anchor = (0,0),borderType=cv2.BORDER_REFLECT)
        filtered_im2 = cv2.filter2D(img2,-1,downsample_filter,anchor = (0,0),borderType=cv2.BORDER_REFLECT)
        img1 = filtered_im1[::2,::2]
        img2 = filtered_im2[::2,::2]

    # print(np.power(mcs_array[:level-1],weight[:level-1]))
    # print(mssim_array[level-1]**weight[level-1])
    overall_mssim = np.prod(np.power(mcs_array[:level-1],weight[:level-1]))*(mssim_array[level-1]**weight[level-1])

    return overall_mssim

def detect(im1,im2):

    kp1, kp2, matches = Compute_ORB(im1, im2)
    warped = Transform_Sansac(im1, im2, kp1, kp2, matches)
    length = 40
    im1, warped = CutEdges(im1, warped, length)
    res = msssim(im1, warped)
    return res, warped

if __name__ == "__main__":
    im11 = cv2.imread(r'E:\03-Anomaly Detection\try\T\naviID_4_devID_1_pos_1_1.jpg', 0)
    im22 = cv2.imread(r'E:\03-Anomaly Detection\try\N\naviID_4_devID_1_pos_1_1.jpg', 0)
    im1 = cv2.resize(im11, (112, 112))
    im2 = cv2.resize(im22, (112, 112))
    res,warped  =  detect(im1, im2)
    plt.imshow(warped), plt.show()

    # plt.imshow(res), plt.show()
    print(res)
    print(type(res))
    print(res.shape())

    print(res.shape())
