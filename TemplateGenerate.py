# -*- coding: utf-8 -*-
# @Time : 2022/4/7 下午20:31
# @Author : iano
# @File : TemplateGenerate.py
# @desc: 通过目标检测和人工补充，生成符合要求的模板文件
import json
import numpy as np
import cv2
import os
import copy
import yolofunction
import random
import colorsys
'''
算法内部使用，不需要生成jason格式
[编号，label, X, Y, W, H, 位置，报警等级]
plot带label+编号显示的结果图，便于人工确认哪个项点需要算法特别关注（升级报警等级）
'''
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

def insertSort(mat):
    list = mat[:, 2]
    for i in range(1,len(list)):
        j = i-1
        key = list[i]
        k = copy.deepcopy(mat[i,:])
        while j >= 0:
            if list[j] > key:
                mat[j+1,:] = mat[j,:]
                mat[j,:]=k
            j -= 1
    return mat

def draw_bbox_new(image, bboxes, filename, classes=None, show_label=True):
    class_num = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / class_num, 1., 1.) for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 230), int(x[1] * 220), int(x[2] * 220)), colors))
    random.seed(32)
    random.shuffle(colors)
    random.seed(None)
    importance = []
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        class_ind = int(bbox[4])
        bbox_color = colors[class_ind]
        bbox_thick = 17
        c1 = (coor[0], coor[1])
        c2 = (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        if show_label:
            bbox_mess = '%d.%s: %.2f' % (i+1, classes[class_ind], 1)
            t_size = cv2.getTextSize(bbox_mess, 1, 6.5, thickness=40)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled
            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 7,
                        lineType=cv2.LINE_AA)
        cv2.namedWindow(filename, 0)
        cv2.resizeWindow(filename, 750, 750)
        cv2.moveWindow(filename, 400, 30)
        cv2.imshow(filename, image)
        cv2.waitKey(10)
        degree = int(input('输入刚刚显示项点的重要等级：'))
        print(degree)
        if degree not in [1,2,3]:
            degree = int(input('输入刚刚显示项点的重要等级(1关键 2一般 3非关键)：'))
        importance.append(degree)
    return importance


path_base = r'E:\04-Data\02-robot\05-Xingqiao0708'
template_path = 'data\\Line3\\StandardDefautTree\\labels\\'
weights = r'C:/Users/Lenovo/PycharmProjects/yolov5-wjd/runs/train/best.pt' # yolo检测权重
# filename_txt = '1.txt'
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
classes_d = ['KeyholeCap', 'Valve', 'RubberPile', 'Sewing'] # 分类网络
classes_e = ['TemperaturePaste', 'Oil', 'SteelWire', 'SplitPin'] # 专项网络
classes_f = ['Grille', 'AirOutlet', 'BrakeShoe'] # 3D网络
classes_g = ['Box']
w=h=5120

base_path = 'E:\\04-Data\\02-robot\\05-Xingqiao0708\\'  # 项目根路径 下有newget template含有原始图片的文件夹及保存路径
path_label_o = 'E:\\04-Data\\02-robot\\05-Xingqiao0708\\others\\completeresults\\labels\\'   # 读取模板label路径

for filename in os.listdir(os.path.join(path_base,'Template')):
    filename = 'T4B11.jpg'
    img_path = os.path.join(path_base, "Template\\", filename) # 模板图片
    img = cv2.imread(img_path)

    # 读取模板 将ground_truth中全部n1个框坐标存入gt_boxes
    filename_txt = filename.replace('.jpg', '.txt')
    matrix = readyololabelfile(path_label_o, filename_txt, w, h)
    gt_boxes_ = np.zeros((matrix.shape[0], 8))
    gt_boxes_[:, 0:6] = matrix
    print('gt_:',gt_boxes_)
    gt_boxes = insertSort(gt_boxes_)  # 项点根据坐标顺序进行排序
    print('gt:',gt_boxes)

    # num label x y w h importance program


    bboxes = []
    for i in range(gt_boxes.shape[0]):
        aa = gt_boxes[i, :]
        print('aa:', aa)
        label = classes[int(aa[1])]
        # print(label)
        x_center = float(aa[2])  # aa[1]左上点的x坐标
        y_center = float(aa[3])  # aa[2]左上点的y坐标
        width = int(float(aa[4]))  # aa[3]图片width
        height = int(float(aa[5]))  # aa[4]图片height
        lefttopx = int(x_center - width / 2.0)
        lefttopy = int(y_center - height / 2.0)
        roi = img[lefttopy:lefttopy + height, lefttopx:lefttopx + width]
        box = [lefttopx, lefttopy, lefttopx + width, lefttopy + height, aa[1]]
        print('box:', box)
        bboxes.append(box)
    # print('bboxes:', bboxes)
    imp = draw_bbox_new(img, bboxes, 'ddd', classes)
    print(imp)
    gt_boxes[:, 6] = imp

                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          
    # # 确定对应的处理算法  ！！！后续单独形成模板生成程序！！！
    # for i in range(matrix.shape[0]):
    #     if classes[int(gt_boxes_[i, 1])] in classes_a:
    #         gt_boxes_[i, 6] = 1
    #     elif classes[int(gt_boxes_[i, 1])] in classes_b:
    #         # print(classes[int(gt_boxes_[i,1])])
    #         gt_boxes_[i, 6] = 2
    #     elif classes[int(gt_boxes_[i, 1])] in classes_c:
    #         gt_boxes_[i, 6] = 3
    #     elif classes[int(gt_boxes_[i, 1])] in classes_d:
    #         gt_boxes_[i, 6] = 4
    #     else:
    #         gt_boxes_[i, 6] = 5
    #
    # file_path = os.path.join(template_path, filename) # 模板图片路径
    # det_boxes = yolofunction.detect(file_path, device, model, half, imgsz, stride)
    #
    # for i in range(len(det_boxes)):
    #
    #     mat = readyololabelfile(path_label_o, filename_txt, w, h)
    # with open(os.path.join(path_label_o, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
    #
    #     n1 = 0
    #     for line in f:
    #         aa = line.split(" ")
    #         a = int(aa[0])
    #         # print(label)
    #         x_center = w * float(aa[1])  # aa[1]左上点的x坐标
    #         y_center = h * float(aa[2])  # aa[2]左上点的y坐标
    #         width = int(w * float(aa[3]))  # aa[3]图片width
    #         height = int(h * float(aa[4]))  # aa[4]图片height
    #         # lefttopx = int(x_center - width / 2.0)
    #         # lefttopy = int(y_center - height / 2.0)
    #         # rightlowx = lefttopx + width
    #         # rightlowy= lefttopy + height
    #
    #         # 确定检修重要等级
    #         d = 2
    #         # 确定项点部位
    #
    #         # 确定状态判别算法
    #         label = classes[a]
    #         if label in classes_a:
    #             gram = 'Detection'
    #         elif label in classes_b or classes_c:
    #             gram = 'MarkLine'
    #         elif label in classes_d:
    #             gram = 'Classify'
    #         else:
    #             gram = 'Others'
    #         # 确定故障类型
    #
    #         b = {"label":str(a)+'-'+classes[a], "CenterX":x_center, "CenterY":y_center, "Width":width, "Height":height,
    #              "lefttopx":lefttopx, "lefttopy":lefttopy, "rightlowx":rightlowx, "rightlowy":rightlowy,
    #              "degree":d, 'C_Gram':gram}
    #
    #         # 项点编号
    #         templateP["P0301204024"+str(n1).zfill(3)] = b
    #         n1 = n1 + 1
    #
    # '''
    # dump: 将数据写入json文件中
    # '''
    # filename_json = filename_txt.replace('txt','json')
    # with open(os.path.join(path_label_o, filename_json), "w") as f:
    #     json.dump(templateP, f)
    #     print("加载入文件完成...")

    # print(templateP['P0301204024001']['label']+':',end='')
    # print(templateP['P0301204024001']['C_Gram'])

# # 读取
# # 将ground_truth中全部n1个框坐标存入gt_boxes
#         filename_json = filename.replace('.bmp', '.json')
#         w = h = 5120
#         with open(os.path.join(path_label_o, filename_json), 'r') as f:
#             load_dict = json.load(f)
#             N = len(load_dict)
#             gt_boxes_ = np.zeros((N, 6))
#             for num in load_dict.keys():
#                 label = load_dict[num]["label"]
#                 x_center = load_dict[num]["CenterX"]  # aa[1]左上点的x坐标
#                 y_center = load_dict[num]["CenterY"]  # aa[2]左上点的y坐标
#                 width = load_dict[num]["Width"]  # aa[3]图片width
#                 height = load_dict[num]["Height"]  # aa[4]图片height
#                 b = [num, label, x_center, y_center, width, height]
#                 print(b)
#                 gt_boxes_[num,:] = b
