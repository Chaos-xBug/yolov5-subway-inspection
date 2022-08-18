# -*- coding: utf-8 -*-
# @Time : 2022/4/6 下午20:31
# @Author : iano
# @File : main02.py
# @desc: 函数化部分功能，ORB特征替换SIFT，改用BF匹配
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
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

import yolofunction
import KeyholeCheck
import SplitpinCheck

from markline.boltdetect import Bolt_State
from markline.cablejointdetect import CableJoint_State
from markline.crossboltdetect import CrossBolt_State
from markline.fasteningdetect import Fastening_State
from markline.groundterminaldetect import GroundTerminal_State
from markline.pipeclampdetect import PipeClamp_State
from markline.unionnutdetect import UnionNut_State


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
        if m.distance < 0.8* n.distance:
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
        matrix_ = np.zeros((300, 6))
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
    w = list[num, 4]
    h = list[num, 5]
    roi = img_check3[int(max(list[num, 3] - h / 2 - extraL, 0)):int(min(list[num, 3] + h / 2 + extraL, 5120)),
          int(max(list[num, 2] - w / 2 - extraL, 0)):int(min(list[num, 2] + w / 2 + extraL, 5120))]

    roi_ = warped[int(max(list[num, 3] - h / 2 - extraL, 0)):int(min(list[num, 3] + h / 2 + extraL, 5120)),
          int(max(list[num, 2] - w / 2 - extraL, 0)):int(min(list[num, 2] + w / 2 + extraL, 5120))]
    return roi, roi_


def roi_write(roi, roi_, filename, label, n, save_cut_path1, save_cut_path2):

    filename_last = filename[:-4] + '_' +  label + str(n) + '.jpg'  # 裁剪出来的小图文件名
    # print(filename_last)
    path1 = os.path.join(save_cut_path1, label)  # 需要在path3路径下创建一个roi文件夹
    path2 = os.path.join(save_cut_path2, label)  # 需要在path3路径下创建一个roi文件夹

    # print('path2:', path2)                    # 裁剪小图的保存位置
    # print(os.path.join(path1, filename_last))
    # print(roi)
    cv2.imwrite(os.path.join(path1, filename_last), roi)
    cv2.imwrite(os.path.join(path2, filename_last), roi_)


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
    path_base = 'E:\\04-Data\\02-robot\\06-Xingqiao0722\\'  # 项目根路径 下有newget template含有原始图片的文件夹及保存路径
    path_label_o = 'E:\\04-Data\\02-robot\\06-Xingqiao0722\\26A(2022-07-22 144554)\\2DResults\\labels\\'   # 读取模板label路径
    path_label_check = 'D:\\pycharmproject\\06-Xingqiao0722\\26A(2022-07-22 161000)\\2DResults\\labels\\'      # 原 读取待检测图像label
    weights = r'C:/Users/Lenovo/PycharmProjects/yolov5-wjd/runs/train/best.pt' # yolo检测权重
    weights_splitpin = r'C:\Users\Lenovo\PycharmProjects\yolov5-wjd\runs\train\expSplitpin\last.pt' # 开口销检测权重
    classes = ['Bolt', 'KeyholeCap', 'SplitPin', 'RubberPlug', 'Valve', 'Oil', 'TemperaturePaste', 'Nameplate',
               'Bandage', 'Hoop',
               'UnionNut', 'PipeClamp', 'CrossBolt', 'BellowsHoop', 'BrakeWireJoint', 'CableJoint', 'GroundTerminal',
               'Fastening', 'WheelTread', 'BrakeShoe',
               'Grille', 'AirOutlet', 'RubberPile', 'Sewing', 'Box', 'RectConnector', 'SteelWire', 'PressureTestPort',
               'CircleConnect',
               'Wheellabel', 'MountingBolt', 'Scupper', 'PullTab', 'PlasticPlug', 'DrainCock', 'Warning',
               'BoxCoverHinge', 'noBolt']
    # 算法分组
    classes_a = ['Warning', 'Bandage', 'Hoop', 'Nameplate', 'BellowsHoop','SteelWire','Scupper', 'PullTab', 'PlasticPlug', 'RubberPlug', 'DrainCock', 'RectConnector', 'BoxCoverHinge'] # 13
    classes_b = ['Bolt', 'UnionNut', 'PipeClamp', 'CrossBolt', 'GroundTerminal', 'CableJoint', 'Fastening', 'CircleConnect'] # 8
    classes_c = ['KeyholeCap', 'SplitPin', 'Valve']  # 专项 3
    classes_d = ['Grille', 'AirOutlet', 'RubberPile', 'Sewing', 'Box', ] # 异物i 5
    classes_e = ['Oil', 'TemperaturePaste', 'BrakeWireJoint', 'WheelTread', 'BrakeShoe','PressureTestPort', 'Wheellabel', 'MountingBolt'] #8

    # 保存小图
    save_cut_path1 = r'E:\04-Data\02-robot\06-Xingqiao0722\cutimages\144554\\'
    save_cut_path2 = r'E:\04-Data\02-robot\06-Xingqiao0722\cutimages\161000\\'
    n0 = 38
    for i in range(n0):
        isExists = os.path.exists(save_cut_path1 + classes[i])
        if not isExists:  # 判断如果文件不存在,则创建
            os.makedirs(save_cut_path1 + classes[i])
            print("%s 目录创建成功" % i)
        else:
            print("%s 目录已经存在" % i)
            continue  # 如果文件不存在,则继续上述操作,直到循环结束
        isExists2 = os.path.exists(save_cut_path2 + classes[i])
        if not isExists2:  # 判断如果文件不存在,则创建
            os.makedirs(save_cut_path2 + classes[i])
            print("%s 目录创建成功" % i)
        else:
            print("%s 目录已经存在" % i)
            continue  # 如果文件不存在,则继续上述操作,直到循环结束

    # 检测模型加载
    check = cp.Siamese()  # 缺失项点分类确认判断
    device, model, half, imgsz, stride = yolofunction.initialze(weights) # yolo检测模型初始化
    device_1, model_1, half_1, imgsz_1, stride_1 = SplitpinCheck.initialze(weights_splitpin)


    # for filename in os.listdir(os.path.join(path_base, "Newget")):
    for filename in os.listdir(os.path.join(path_base, "26A(2022-07-22 144554)\\2D")):
        print(filename)
        # filename = 'Pic_2022_05_12_150833_46.jpg'
        path1 = os.path.join(path_base, "26A(2022-07-22 144554)\\2D\\", filename)  # 模板图片
        print(path1)
        path2 = os.path.join(path_base, "26A(2022-07-22 161000)\\2D\\", filename)  # 原始待检测图片
        # path1 = os.path.join(path_base, "Template\\", filename) # 模板图片
        # path2 = os.path.join(path_base, "Newget\\", filename) # 原始待检测图片
        save_path = os.path.join(path_base, "manques\\", filename) # 缺失项点截取图像
        Splitpin_save_path = os.path.join(path_base, "SplitpinParts\\", filename) # 缺失项点截取图像
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
        # s0 = time.time()
        warped, Minv = template_match(img_check3, img_o3) # 变形图像 变换矩阵
        # print('匹配耗时{:.3f}秒'.format(time.time()-s0))
        # matplotlib.use('TkAgg')
        plt.imshow(img_o3), plt.title('Standard Image'), plt.show()
        plt.imshow(warped), plt.title('Warped Standard Image'), plt.show()

        # 读取模板 将ground_truth中全部n1个框坐标存入gt_boxes
        filename_txt = filename.replace('.jpg', '.txt')
        matrix = readyololabelfile(path_label_o, filename_txt, w, h)
        gt_boxes_ = np.zeros((matrix.shape[0], 7))
        gt_boxes_[:,0:6] = matrix
        # num laebl x y w h program
        # 确定对应的处理算法  ！！！后续单独形成模板生成程序！！！
        for i in range(matrix.shape[0]):
            if classes[ int(gt_boxes_[i,1])] in classes_a:
                gt_boxes_[i, 6] = 1
            elif classes[int(gt_boxes_[i,1])] in classes_b:
                # print(classes[int(gt_boxes_[i,1])])
                gt_boxes_[i, 6] = 2
            elif classes[int(gt_boxes_[i,1])] in classes_c:
                gt_boxes_[i, 6] = 3
            elif classes[int(gt_boxes_[i,1])] in classes_d:
                gt_boxes_[i, 6] = 4
            else:
                gt_boxes_[i, 6] = 5
        # print(gt_boxes_)
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
        uncoverlist_ = np.zeros(gt_boxes.shape)
        count1 = 0
        coverlist_ = np.zeros(gt_boxes.shape)
        count2 = 0
        for i in range(gt_boxes.shape[0]):
            if gt_boxes[i, 2] < 30 or gt_boxes[i, 2] > 5090 or gt_boxes[i, 3] < 30 or gt_boxes[i, 3] > 5090: # 应该添加w h
                uncoverlist_[count1, :] = gt_boxes[i, :]
                count1 = count1 + 1
            else: #
                coverlist_[count2, :] = gt_boxes[i, :]
                count2 = count2 + 1
        uncoverlist = uncoverlist_[0:count1, :]
        coverlist = coverlist_[0:count2, :]
        # print('gt_boxes个数:', gt_boxes.shape[0])
        print('coverlist个数:', coverlist.shape[0])
        print('uncoverlist个数:', uncoverlist.shape[0])
        '''
        Step2 目标检测
        依次拿检测到的det_boxes和真值ground truth中gt_box, 对gt_box坐标进行匹配变换，计算变换后框图与det_boxes的IOU
        确定无法覆盖项点列表、检测缺失项点列表
        '''
        # 将detected的全部n2个框坐标存入det_boxes
        # det_boxes = readyololabelfile(path_label_check, filename_txt, w, h)

        det_boxes = yolofunction.detect(path2, device, model, half, imgsz, stride)

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
            # print(result)
            # catimage = np.concatenate((roi, roi_))
            # cv2.imshow(classes[int(lostlist_[i, 1])], catimage)
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
        # device, model, half, imgsz, stride = SplitpinCheck.initialze(weights)
        for i in range(len(matchlist)):
            label = classes[int(matchlist[i, 1])]
            roi, roi_ = cutlocalimg(matchlist, i, 15)
            roi_write(roi, roi_, filename, label, i, save_cut_path1, save_cut_path2)


        # for i in range(len(matchlist)):
        #     label = classes[int(matchlist[i, 1])]
        #     roi, roi_ = cutlocalimg(matchlist, i, 15)
        #     roi_write(roi,roi_,filename,label, i)
        #     if matchlist[i, 6] == 1:
        #         roi, roi_ = cutlocalimg(matchlist, i, 15)
        #         print('只检测缺失：', classes[int(matchlist[i, 1])])
        #         catimage = np.concatenate((roi, roi_))
        #         # print(matchlist[i, :])
        #         cv2.imshow("存在项", catimage)
        #         cv2.waitKey(0)
        #         continue
        #     elif matchlist[i, 6] == 2:
        #         # 标记线检测算法
        #         roi, roi_ = cutlocalimg(matchlist, i, 15)
        #         # 0:Bolt, 15:CableJoint, 12:CrossBolt, 17:Fastening, 16:GroundTerminal, 11:PipeClamp, 10:UnionNut
        #         if matchlist[i, 1] == 0:
        #             state, old_img = Bolt_State(roi)
        #         elif matchlist[i, 1] == 15:
        #             state, old_img = CableJoint_State(roi)
        #         elif matchlist[i, 1] == 12:
        #             state, old_img = CrossBolt_State(roi)
        #         elif matchlist[i, 1] == 17:
        #             state, old_img = Fastening_State(roi)
        #         elif matchlist[i, 1] == 16:
        #             state, old_img = GroundTerminal_State(roi)
        #         elif matchlist[i, 1] == 11:
        #             state, old_img = PipeClamp_State(roi)
        #         elif matchlist[i, 1] == 10:
        #             state, old_img = UnionNut_State(roi)
        #         else:
        #             state = -3 #'BrakeWireJoint'
        #         catimage = np.concatenate((roi, roi_))
        #         print('标记线检测状态为：', state)
        #         # print(matchlist[i, :])
        #         cv2.imshow("markline", catimage)
        #         cv2.waitKey(0)
        #     elif matchlist[i, 6] == 3: # 专项算法
        #         if matchlist[i, 1] == 1:
        #             # 锁孔状态检查'KeyholeCap'
        #             roi, roi_ = cutlocalimg(matchlist, i, 15)
        #             state = KeyholeCheck.KeyholeState(roi)
        #             defautlist[i, 0:7] = matchlist[i, :]
        #             defautlist[i, 7] = state
        #             print('锁孔状态为：', state)
        #             catimage = np.concatenate((roi, roi_))
        #             # print(matchlist[i, :])
        #             cv2.imshow("keyhole", catimage)
        #             cv2.waitKey(0)
        #         elif matchlist[i, 1] == 2:
        #             # 开口销状态检查'SplitPin'
        #             roi, roi_ = cutlocalimg(matchlist, i, 15)
        #             cv2.imwrite(Splitpin_save_path, roi)
        #             # source = os.path.join(Splitpin_save_path, r"Test\\", filename)
        #             res, img = SplitpinCheck.detect(Splitpin_save_path, device_1, model_1, half_1, imgsz_1, stride_1)
        #             print('开口销状态为:', res)
        #             catimage = np.concatenate((roi, roi_))
        #             # print(matchlist[i, :])
        #             cv2.imshow("Splitpin", catimage)
        #             cv2.waitKey(0)
        #         elif matchlist[i, 1] == 2:
        #             # 开关状态检查 Valve'
        #             state = 1
        #     elif matchlist[i, 6] == 4: # 异物检测
        #         roi, roi_ = cutlocalimg(matchlist, i, 15)
        #         catimage = np.concatenate((roi, roi_))
        #         print('异物检测待开发 label:', classes[int(matchlist[i, 1])])
        #         # print(matchlist[i, :])
        #         cv2.imshow("others", catimage)
        #         cv2.waitKey(0)
        #     else: # 不用检测
        #         roi, roi_ = cutlocalimg(matchlist, i, 15)
        #         catimage = np.concatenate((roi, roi_))
        #         print('暂无需检测 label:',classes[int(matchlist[i, 1])])
        #         # print(matchlist[i, :])
        #         cv2.imshow("others", catimage)
        #         cv2.waitKey(0)