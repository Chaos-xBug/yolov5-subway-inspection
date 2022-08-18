# -*- coding: utf-8 -*-
# @Time : 2022/3/29 上午11:31
# @Author : iano
# @File : main.py
# @desc:
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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
        # matchesMask = mask.ravel().tolist()  # ravel()展平，并转成列表

        h, w = image1.shape
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

def compute_iou(box1, box2, wh=True):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1])) # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6) #计算交并比

    return iou


if __name__ == "__main__":
    '''
    批量检测
    '''
    count = 0
    path_base = 'data\\Line3\\'
    filename = '1.bmp'
    # for filename in os.listdir(os.path.join(path_base, "Defaut")):
    count = count + 1
    path1 = os.path.join(path_base, "Standard\\", filename) # 标准图片
    path2 = os.path.join(path_base, "Defaut\\", filename) # 原始待检测图片
    classes = ['Bolt', 'KeyholeCap', 'SplitPin', 'RubberPlug', 'Valve', 'Oil', 'TemperaturePaste', 'Nameplate',
               'Bandage', 'Hoop',
               'UnionNut', 'PipeClamp', 'CrossBolt', 'BellowsHoop', 'BrakeWireJoint', 'CableJoint', 'GroundTerminal',
               'Fastening', 'WheelTread', 'BrakeShoe',
               'Grille', 'AirOutlet', 'RubberPile', 'Sewing', 'Box', 'RectConnector', 'SteelWire', 'PressureTestPort',
               'CircleConnect',
               'Wheellabel', 'MountingBolt', 'Scupper', 'PullTab', 'PlasticPlug', 'DrainCock', 'Warning',
               'BoxCoverHinge', 'noBolt']

    '''
    Step1 模板匹配
    '''
    img_o = cv2.imread(path1, 0)
    img_check = cv2.imread(path2, 0)
    m, n = img_o.shape
    kp1, kp2, matches = Compute_Sift(img_o, img_check)
    warped, Minv= Transform_Sansac(img_o, img_check, kp1, kp2, matches)
    # try:
    #     if warped == -1:
    #         continue
    # except:
    #     print(count)
    plt.imshow(img_o, 'gray'), plt.title('Standard Image'), plt.show()
    plt.imshow(warped, 'gray'), plt.title('Warped Test Image'), plt.show()

    # 确认未覆盖项点及无对比项点
    # img1_mask = functions.Add_Mask_2(img_o, coordinates)   # 掩膜处理
    # # plt.imshow(img1_mask, 'gray'), plt.title('Standart Image with mask'), plt.show()

    '''
    Step2 目标检测
    '''
    # 依次拿检测到的det_boxes和真值ground truth中gt_box匹配，计算两者IOU
    path_label_o = 'data\\Line3\\StandardDefautTree\\labels\\'
    path_label_check = 'data\\Line3\\DefautTestResult\\labels\\'
    filename_txt = filename.replace('.bmp', '.txt')
    w = h = 5120
    # 将ground_truth中全部n1个框坐标存入gt_boxes
    with open(os.path.join(path_label_o, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
        matrix = np.zeros((200, 4))
        label = []
        n1 = 0
        for line in f:
            aa = line.split(" ")
            a = str(aa[0])
            # print(label)
            x_center = w * float(aa[1])  # aa[1]左上点的x坐标
            y_center = h * float(aa[2])  # aa[2]左上点的y坐标
            width = int(w * float(aa[3]))  # aa[3]图片width
            height = int(h * float(aa[4]))  # aa[4]图片height
            b = [x_center, y_center, width, height]
            # print(b)
            matrix[n1, :] = b
            label.append(aa[0])
            n1 = n1 + 1
        gt_boxes = matrix[0:n1, :]

    # 将detected的全部n2个框坐标存入det_boxes
    with open(os.path.join(path_label_check, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
        matrix = np.zeros((200, 4))
        n2 = 0
        for line in f:
            aa = line.split(" ")
            a = str(aa[0])
            # print(label)
            x_center = w * float(aa[1])  # aa[1]左上点的x坐标
            y_center = h * float(aa[2])  # aa[2]左上点的y坐标
            width = int(w * float(aa[3]))  # aa[3]图片width
            height = int(h * float(aa[4]))  # aa[4]图片height
            b = [x_center, y_center, width, height]
            # print(b)
            matrix[n2, :] = b
            n2 = n2 + 1
        det_boxes_ = matrix[0:n2, :]

    # 通过矩阵变换到模板图片中对应的位置
    det_boxes = np.zeros(det_boxes_.shape)
    det_boxes[:,2:4] = det_boxes_[:,2:4]
    for i in range(det_boxes_.shape[0]):
        pts_ = np.float32(det_boxes_[i, 0:2]).reshape(-1, 1, 2)
        ddd = cv2.perspectiveTransform(pts_, Minv)  # image2变换后的顶点坐标
        det_boxes[i, 0:2] = ddd.reshape(1, 2)

        # 通过画图表示前后变换的框图位置对比验证
        w = det_boxes_[i, 2]
        h = det_boxes_[i, 3]
        cv2.rectangle(img_check, (int(det_boxes_[i, 0] - w / 2), int(det_boxes_[i, 1] - h / 2)),
                      (int(det_boxes_[i, 0] + w / 2), int(det_boxes_[i, 1] + h / 2)), (0, 0, 255), 10)
        cv2.rectangle(warped, (int(det_boxes[i, 0] - w / 2), int(det_boxes[i, 1] - h / 2)),
                      (int(det_boxes[i, 0] + w / 2), int(det_boxes[i, 1] + h / 2)), (0, 0, 255), 10)
    # 画图检查
    plt.imshow(img_check, 'gray'), plt.title('img_check with boxes'), plt.show()
    plt.imshow(warped, 'gray'), plt.title('warped with boxes'), plt.show()
    # cv2.imwrite('img_check.jpg', img_check)
    # cv2.imwrite('warped.jpg', warped)
    # 画标准图框图进行对比
    for i in range(gt_boxes.shape[0]):
        w = gt_boxes[i, 2]
        h = gt_boxes[i, 3]
        cv2.rectangle(img_o, (int(gt_boxes[i, 0] - w / 2), int(gt_boxes[i, 1] - h / 2)),
                      (int(gt_boxes[i, 0] + w / 2), int(gt_boxes[i, 1] + h / 2)), (0, 0, 255), 10)
    plt.imshow(img_o, 'gray'), plt.title('img_o with boxes'), plt.show()


    # IOU计算如下，一共n1*n2个,第m行是第m个det_box和n1个gt_box的IOU
    IOU = np.zeros((n2, n1))
    for i in range(n2):
        for j in range(n1):
            IOU[i, j] = compute_iou(det_boxes[i, :], gt_boxes[j, :])


    # 找出每个det_box和三个gt_box IOU最值
    iou_max = np.max(IOU, axis=1)
    print('iou_max.shape:', iou_max.shape)
    print('iou.shape:', IOU.shape)

    # 对IOU进行过滤小于0.5淘汰
    iou_max_index = iou_max > 0.5
    print(iou_max_index)
    # iou_max_index=[ True  True False  True]
    new_det_boxes = det_boxes[iou_max_index]
    # [list([1, 2, 4, 5]) list([4, 5, 6, 7]) list([8, 9, 10, 11, 12])
    print('new_det_boxes.shape:', new_det_boxes.shape)
    # 已经过滤掉det_box的第四个box
    # 这时候需要给每个det_box 匹配对应的gt_box,首先去除IOU中那个属于错误检测框的IOU
    gt_match_dt_index = np.argmax(IOU, axis=0)  # 返回给gt_box匹配IOU最大的对应det_box的索引
    print('gt_match_dt_index:', gt_match_dt_index)  # [1 0 2]

    iou_max = np.max(IOU, axis=0)  # 按模板寻找匹配结果
    iou_max_index = iou_max > 0.4
    new_gt_boxes = gt_boxes[iou_max_index]  # 确认那些模板已被找到
    # 将未匹配到结果的模板在待测照片对应位置中截取
    for i in range(len(iou_max)):
        if iou_max_index[i] == False:
            w = gt_boxes[i, 2]
            h = gt_boxes[i, 3]
            roi = img_o[int(gt_boxes[i, 1] - h / 2):int(gt_boxes[i, 1] + h / 2), int(gt_boxes[i, 0] - w / 2):int(gt_boxes[i, 0] + w / 2)]
            plt.imshow(roi, 'gray'), plt.title(classes[int(label[i])]), plt.show()
            roi_ = warped[int(gt_boxes[i, 1] - h / 2):int(gt_boxes[i, 1] + h / 2),
                  int(gt_boxes[i, 0] - w / 2):int(gt_boxes[i, 0] + w / 2)]
            plt.imshow(roi_, 'gray'), plt.title(classes[int(label[i])]), plt.show()





