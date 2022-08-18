# -*- coding: utf-8 -*-
# @Time : 2022/4/6 下午20:31
# @Author : iano
# @File : main02.py
# @desc: 函数化部分功能，ORB特征替换SIFT，改用BF匹配
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
from PIL import Image
import json
from markline import predict as mp
from classfier import predict as cp
import time
import ssim
import ms_ssim1

import KeyholeCheck
import BoltMarline

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

def template_match(image1, image2):

    kp1, kp2, matches = Compute_ORB(image1, image2)
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

def readyololabelfile(path_label_o, filename_txt, w, h):
    with open(os.path.join(path_label_o, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
        matrix_ = np.zeros((200, 6))
        n1 = 0
        for line in f:
            aa = line.split(" ")
            a = int(aa[0])
            # print(label)
            x_center = w * float(aa[1])  # aa[1]左上点的x坐标
            y_center = h * float(aa[2])  # aa[2]左上点的y坐标
            width = int(w * float(aa[3]))  # aa[3]图片width
            height = int(h * float(aa[4]))  # aa[4]图片height
            b = [n1, a, x_center, y_center, width, height]
            # [编号，label, X, Y, W, H]
            matrix_[n1, :] = b
            n1 = n1 + 1
        matrix = matrix_[0:n1, :]
        return matrix

def plotrectangle(boxes,image):
    for i in range(boxes.shape[0]):
        l = boxes[i, 4]
        m = boxes[i, 5]
        cv2.rectangle(image, (int(boxes[i, 2] - l / 2), int(boxes[i, 3] - m / 2)),
                      (int(boxes[i, 2] + l / 2), int(boxes[i, 3] + m / 2)), (0, 0, 255), 10)
    # # 画图检查
    plt.imshow(image), plt.title('image with boxes'), plt.show()
    # cv2.imwrite('img_check.jpg', image)

def cutlocalimg(list, num, extraL):
    w = matchlist[num, 4]
    h = matchlist[num, 5]
    roi = img_check3[int(max(list[num, 3] - h / 2 - extraL, 0)):int(min(list[num, 3] + h / 2 + extraL, 5120)),
          int(max(list[num, 2] - w / 2 - extraL, 0)):int(min(list[num, 2] + w / 2 + extraL, 5120))]

    roi_ = warped[int(max(list[num, 3] - h / 2 - extraL, 0)):int(min(list[num, 3] + h / 2 + extraL, 5120)),
          int(max(list[num, 2] - w / 2 - extraL, 0)):int(min(list[num, 2] + w / 2 + extraL, 5120))]
    return roi, roi_

class oilstate:
    '所有项点的基类'

    def __init__(self, lable, centerX, centerY, boxW, boxH):
        self.lable = lable
        self.centerX = centerX
        self.centerY = centerY
        self.boxW = boxW
        self.boxH = boxH

    def DiagonalPiont(self):
        point1 = [int(self.centerX - self.boxW / 2), int(self.centerY - self.boxH  / 2)]
        point2 = [int(self.centerX + self.boxW / 2), int(self.centerY + self.boxH  / 2)]
        return point1, point2

    def Statecheck(self):
        """"
        油液位检测方法及结果输出
        """


if __name__ == "__main__":
    '''
    初始化定义
    '''
    path_base = 'D:\\pycharmproject\\yolov5-wjd\\data\\Line3\\'
    path_label_o = 'D:\\pycharmproject\\yolov5-wjd\\data\\Line3\\StandardDefautTree\\labels\\'
    path_label_check = 'D:\\pycharmproject\\yolov5-wjd\\data\\Line3\\DefautTestResult\\labels\\'
    classes = ['Bolt', 'KeyholeCap', 'SplitPin', 'RubberPlug', 'Valve', 'Oil', 'TemperaturePaste', 'Nameplate',
               'Bandage', 'Hoop',
               'UnionNut', 'PipeClamp', 'CrossBolt', 'BellowsHoop', 'BrakeWireJoint', 'CableJoint', 'GroundTerminal',
               'Fastening', 'WheelTread', 'BrakeShoe',
               'Grille', 'AirOutlet', 'RubberPile', 'Sewing', 'Box', 'RectConnector', 'SteelWire', 'PressureTestPort',
               'CircleConnect',
               'Wheellabel', 'MountingBolt', 'Scupper', 'PullTab', 'PlasticPlug', 'DrainCock', 'Warning',
               'BoxCoverHinge', 'noBolt']
    # 算法分组
    classes_a = ['Warning', 'Bandage', 'Hoop', 'Nameplate', 'BellowsHoop']
    classes_b = ['Bolt', 'UnionNut', 'PipeClamp', 'CrossBolt']
    classes_c = ['BrakeWireJoint', 'GroundTerminal', 'Cablejoint', 'Fastening']
    classes_d = ['KeyholeCap', 'Valve', 'RubberPile', 'Sewing']  # 分类网络
    classes_e = ['TemperaturePaste', 'Oil', 'SteelWire', 'SplitPin']  # 专项网络
    classes_f = ['Grille', 'AirOutlet', 'BrakeShoe']  # 3D网络
    classes_g = ['Box']

    # 检测模型加载
    check = cp.Siamese()  # 缺失项点分类确认判断

    for filename in os.listdir(os.path.join(path_base, "Defaut")):
        path1 = os.path.join(path_base, "Standard\\", filename) # 模板图片
        path2 = os.path.join(path_base, "Defaut\\", filename) # 原始待检测图片
        save_path = os.path.join(path_base, "manques\\", filename) # 缺失项点截取图像
        isExists = os.path.exists(save_path)
        if not isExists:  # 判断如果文件不存在,则创建
            os.makedirs(save_path)
            print("目录创建成功")
        else:
            print("目录已经存在")
        '''
        Step1 模板匹配
        将模板与待检图像进行匹配，并根据匹配结果确定哪些项点没有被拍摄到
        '''
        img_o3 = cv2.imread(path1, 1)
        img_check3 = cv2.imread(path2, 1)
        [w, h] = img_o3.shape[0:2]
        s0 = time.time()
        warped, Minv = template_match(img_check3, img_o3) # 变形图像 变换矩阵
        print('匹配耗时{:.3f}秒'.format(time.time()-s0))

        plt.imshow(img_o3), plt.title('Standard Image'), plt.show()
        plt.imshow(warped), plt.title('Warped Standard Image'), plt.show()

        # 读取模板 将ground_truth中全部n1个框坐标存入gt_boxes
        filename_txt = filename.replace('.bmp', '.txt')
        matrix = readyololabelfile(path_label_o, filename_txt, w, h)
        gt_boxes_ = np.zeros((matrix.shape[0], 7))
        gt_boxes_[:,0:6] = matrix
        # num laebl x y w h program
        # 确定对应的处理算法  ！！！后续单独形成模板生成程序！！！
        for i in range(matrix.shape[0]):
            if classes[ int(gt_boxes_[i,1])] in classes_a:
                gt_boxes_[i, 6] = 1
            elif classes[int(gt_boxes_[i,1])] in classes_b or classes_c:
                gt_boxes_[i, 6] = 2
            elif classes[int(gt_boxes_[i,1])] in classes_d:
                gt_boxes_[i, 6] = 3
            else:
                gt_boxes_[i, 6] = 4

        # 模板中项点位置通过矩阵变换到待检图片中对应的位置
        gt_boxes = copy.deepcopy(gt_boxes_)
        for i in range(gt_boxes_.shape[0]):
            pts_ = np.float32(gt_boxes_[i, 2:4]).reshape(-1, 1, 2)
            ddd = cv2.perspectiveTransform(pts_, Minv)  # 坐标变换
            gt_boxes[i, 2:4] = ddd.reshape(1, 2)

        #通过画图表示前后变换的框图位置对比验证
        # plotrectangle(gt_boxes_, img_o3)
        # plotrectangle(gt_boxes, warped)

        '''
        # 确定无法覆盖项点清单
        '''
        # 判断准则，模板变换warped后，项点中心点坐标小于0+30 或者 大于5210-30时
        uncoverlist = np.zeros(gt_boxes.shape)
        count1 = 0
        coverlist = np.zeros(gt_boxes.shape)
        count2 = 0
        for i in range(gt_boxes.shape[0]):
            if gt_boxes[i, 2] < 30 or gt_boxes[i, 2] > 5090 or gt_boxes[i, 3] < 30 or gt_boxes[i, 3] > 5090: # 应该添加w h
                uncoverlist[count1, :] = gt_boxes[i, :]
                count1 = count1 + 1
            else: #
                coverlist[count2, :] = gt_boxes[i, :]
                count2 = count2 + 1
        uncoverlist = uncoverlist[0:count1, :]
        coverlist = coverlist[0:count2, :]
        # print('gt_boxes个数:', gt_boxes.shape[0])
        print('coverlist个数:', coverlist.shape[0])
        print('uncoverlist个数:', uncoverlist.shape[0])
        '''
        Step2 目标检测
        依次拿检测到的det_boxes和真值ground truth中gt_box, 对gt_box坐标进行匹配变换，计算变换后框图与det_boxes的IOU
        确定无法覆盖项点列表、检测缺失项点列表
        '''
        # 将detected的全部n2个框坐标存入det_boxes
        det_boxes = readyololabelfile(path_label_check, filename_txt, w, h)
        # # 画标准图框图进行对比
        # plotrectangle(det_boxes, img_check3)

        '''
        IOU匹配 确定匹配清单和缺失清单
        '''
        # 通过计算IOU对det_boxes和gt_boxes进行匹配
        # IOU计算如下，一共n1*n2个,第m行是第m个det_box和n1个gt_box的IOU
        n1 = coverlist.shape[0]
        n2 = det_boxes.shape[0]
        IOU = np.zeros((n2, n1))
        for i in range(n2):
            for j in range(n1):
                IOU[i, j] = compute_iou(det_boxes[i, 2:6], coverlist[j, 2:6])

        # 找出每个gt_box和全部det_box IOU最值
        iou_max = np.max(IOU, axis=0)
        # print('iou_max', iou_max)
        # 对IOU进行过滤小于0.5淘汰
        iou_max_index = iou_max > 0.4
        # print('iou_max_index:',iou_max_index)
        # iou_max_index=[ True  True False  True]
        matchlist = coverlist[iou_max_index]  # 确认哪些模板已被找到
        lostlist_ = coverlist[iou_max <= 0.4]
        # print(np.shape(IOU))
        print("matchlist_:",len(matchlist))
        print('lostlist_:',len(lostlist_))
        # print('new_gt_boxes:', new_gt_boxes)
        # 给每个gt_box 匹配对应的det_box,首先去除IOU中那个属于错误检测框的IOU
        # gt_match_dt_index = np.argmax(IOU, axis=0)  # 返回给det_box匹配IOU最大的对应gt_box的索引
        # print('gt_match_dt_index:', gt_match_dt_index)  # [1 0 2]
        '''
        对缺失项点进行重复确认
        '''
        lostlist = np.zeros(np.shape(lostlist_))
        count = 0
        extraL = 50
        for i in range(len(lostlist_)):
            roi, roi_ = cutlocalimg(lostlist_, i, extraL)
            '''
            ssim特征确认 
            '''
            # im11= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # im22 = cv2.cvtColor(roi_, cv2.COLOR_BGR2GRAY)
            #
            # im1 = cv2.resize(im11, (224, 224))
            # im2 = cv2.resize(im22, (224, 224))
            # # ssim比对
            # ssim_value, ssim_map = ssim.compute_ssim(np.array(im1), np.array(im2), win_size=16)
            # plt.imshow(ssim_map, 'gray'), plt.show()
            # index = np.unravel_index(ssim_map.argmax(), ssim_map.shape)
            # print(ssim_value)

            # ms-ssim比对
            # res, im1w, im2w = ms_ssim1.detect(im1, im2)
            # print(res)
            #
            # catimage = np.concatenate((im1w, im2w))
            # cv2.imshow("out", catimage)
            # cv2.waitKey(0)

            # row, col = ssim.getMatrixMinNumIndex(ssim_map, 1300)
            # plt.imshow(ssim_map, 'gray')
            # plt.plot(col, row, 'o')
            # plt.show()
            # cv2.imshow("out", ssim_map)
            # cv2.waitKey(0)
            '''
            分类网络确认
            '''
            Imgroi = Image.fromarray(np.uint8(roi))
            Imgroi_ = Image.fromarray(np.uint8(roi_))
            result = check.detect_image(Imgroi, Imgroi_)

            if result < 0.4:
                lostlist[count, :] = lostlist_[i, :]
                count = count + 1
            else:
                matchlist = np.concatenate((matchlist, [lostlist_[i, :]]))
                print('match!')
            print(result)
            # catimage = np.concatenate((roi, roi_))
            # cv2.imshow("out", catimage)
            # cv2.waitKey(0)

        lostlist = lostlist[0:count, :]
        print("matchlist:", len(matchlist))
        print('lostlist:', len(lostlist))

        '''
        Step3 项点状态检测
        根据不同项点类型，利用对应算法模块进行状态判别
        '''
        # 循环matchlist 检查不同项点
        defautlist = np.zeros([100, 7])
        for i in range(len(matchlist)):
            if matchlist[i, 6] == 1:
                continue
            elif matchlist[i, 6] == 2:
                # 标记线检测算法
                roi, roi_ = cutlocalimg(matchlist, i, 15)
                state, old_img = BoltMarline.line_state(roi)
                # catimage = np.concatenate((roi, roi_))
                cv2.imshow("out", old_img)
                cv2.waitKey(0)

            elif matchlist[i, 1] == 1:
                # 锁孔状态检查'KeyholeCap'
                roi, roi_ = cutlocalimg(matchlist, i, 15)
                State = KeyholeCheck.KeyholeState(roi)
                defautlist[i, 0:7] = matchlist[i, :]
                defautlist[i, 7] = State
            elif matchlist[i, 1] == 2:
                # 开口销状态检查'SplitPin'
                roi, roi_ = cutlocalimg(matchlist, i, 15)
            else:
                print(matchlist[i, 1])