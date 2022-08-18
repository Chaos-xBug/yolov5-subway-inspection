'''
对图像内检测的目标进行位置关系上的排列
思路：建立两个数组，从左上角开始寻找点，按横坐标x顺序排列，在寻找下一个点时，若纵坐标在

'''
# from desmonstration.yoloplot import draw_bbox_new
from UniDetection import readyololabelfile
import numpy as np
import colorsys
import random
import copy
import cv2
import os

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
    return mat, list

def draw_bbox_new(image, bboxes, filename, classes=None, show_label=True):
    class_num = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / class_num, 1., 1.) for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 230), int(x[1] * 220), int(x[2] * 220)), colors))
    random.seed(32)
    random.shuffle(colors)
    random.seed(None)
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        class_ind = int(bbox[4])
        bbox_color = colors[class_ind]
        bbox_thick = 7
        c1 = (coor[0], coor[1])
        c2 = (coor[2], coor[3])
        cv2.rectangle(image, c1, c2, bbox_color, bbox_thick)
        if show_label:
            bbox_mess = '%s: %.2f' % (classes[class_ind], 1)
            t_size = cv2.getTextSize(bbox_mess, 0, 0.5, thickness=1)[0]
            cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), bbox_color, -1)  # filled
            cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1,
                        lineType=cv2.LINE_AA)
        cv2.namedWindow(filename, 0)
        cv2.resizeWindow(filename, 750, 750)
        cv2.moveWindow(filename, 400, 30)
        cv2.imshow(filename, image)
        cv2.waitKey(400)

    return image
classes = ['Bolt', 'KeyholeCap', 'SplitPin', 'RubberPlug', 'Valve', 'Oil', 'TemperaturePaste', 'Nameplate',
           'Bandage', 'Hoop',
           'UnionNut', 'PipeClamp', 'CrossBolt', 'BellowsHoop', 'BrakeWireJoint', 'CableJoint', 'GroundTerminal',
           'Fastening', 'WheelTread', 'BrakeShoe',
           'Grille', 'AirOutlet', 'RubberPile', 'Sewing', 'Box', 'RectConnector', 'SteelWire', 'PressureTestPort',
           'CircleConnect',
           'Wheellabel', 'MountingBolt', 'Scupper', 'PullTab', 'PlasticPlug', 'DrainCock', 'Warning',
           'BoxCoverHinge', 'noBolt']

path_label_o = 'E:\\04-Data\\02-robot\\05-Xingqiao0708\\test1-detectresults\\labels\\'   # 读取模板label路径
filename_txt = 'T2B6.txt'
w = h = 5120
matrix = readyololabelfile(path_label_o, filename_txt, w, h)
gt_boxes_ = np.zeros((matrix.shape[0], 7))
gt_boxes_[:,0:6] = matrix
print(gt_boxes_)
print('')
m, l = insertSort(gt_boxes_)
print('m:', m)
print('l:', l)
imgpath = r'E:\04-Data\02-robot\05-Xingqiao0708\Template\T2B6.jpg'
img = cv2.imread(imgpath)
bboxes = []
for i in range(m.shape[0]):
    aa = m[i,:]
    print('aa:',aa)
    label = classes[ int(aa[1])]
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
print('bboxes:',bboxes)
new_image = draw_bbox_new(img, bboxes, 'ddd', classes)
