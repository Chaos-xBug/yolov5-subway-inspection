import random
import colorsys
import matplotlib.pyplot as plt
from datetime import datetime
import os
import cv2
import numpy as np


path_txt = r"E:\04-Data\02-robot\04-Xingqiao0512\desmonstration\labels"         # jpg图片和对应的生成结果的txt标注文件，放在一起
path_image = r"E:\04-Data\02-robot\04-Xingqiao0512\desmonstration\Pics"                  # 测试图片路径
save_path = r'E:\04-Data\02-robot\04-Xingqiao0512\desmonstration\plotresults'  # 裁剪出来的小图保存的根目录

classes = ['Bolt', 'KeyholeCap', 'SplitPin', 'RubberPlug', 'Valve', 'Oil', 'TemperaturePaste', 'Nameplate',
           'Bandage', 'Hoop',
           'UnionNut', 'PipeClamp', 'CrossBolt', 'BellowsHoop', 'BrakeWireJoint', 'CableJoint', 'GroundTerminal',
           'Fastening', 'WheelTread', 'BrakeShoe',
           'Grille', 'AirOutlet', 'RubberPile', 'Sewing', 'Box', 'RectConnector', 'SteelWire', 'PressureTestPort',
           'CircleConnect',
           'Wheellabel', 'MountingBolt', 'Scupper', 'PullTab', 'PlasticPlug', 'DrainCock', 'Warning',
           'BoxCoverHinge', 'noBolt']


def draw_bbox_new(image, bboxes, filename, classes=None, show_label=True):
    class_num = len(classes)
    image_h, image_w, _ = image.shape
    hsv_tuples = [(1.0 * x / class_num, 1., 1.) for x in range(class_num)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 230), int(x[1] * 220), int(x[2] * 220)), colors))
    random.seed(32)
    random.shuffle(colors)
    random.seed(None)
    # print(bboxes)
    print(len(bboxes))
    for i, bbox in enumerate(bboxes):
        coor = np.array(bbox[:4], dtype=np.int32)
        print(classes)
        print(image)
        print(classes)
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
        cv2.waitKey(7)

    return image


def read_class_names(class_file_name):
    names = {}
    ID = 0
    with open(class_file_name, 'r') as data:
        for name in data:
            names[ID] = name.strip('\n')
            ID = ID + 1
    return names


for(root, files, filenames) in os.walk(path_image):    # 遍历文件名
    # print(filename)
    print('first circle')

    for filename in filenames:
        print('filename_img:', filename)
        path1 = os.path.join(path_image, str(filename))
        # print(path1)
        img = cv2.imread(path1)
        h = img.shape[0]
        w = img.shape[1]
        # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)        # resize 图像大小，否则roi区域可能会报错
        filename_txt = filename.replace('.bmp', '.txt')
        # print('filename_txt:', filename_txt)
        n = 1
        bboxes = []

        with open(os.path.join(path_txt, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
            for line in f:
                aa = line.split(" ")
                a = str(aa[0])
                label = classes[int(a)]  # 裁剪出来的小图保存的根目录
                # print(label)
                x_center = w * float(aa[1])       # aa[1]左上点的x坐标
                y_center = h * float(aa[2])       # aa[2]左上点的y坐标
                width = int(w*float(aa[3]))       # aa[3]图片width
                height = int(h*float(aa[4]))      # aa[4]图片height
                lefttopx = int(x_center-width/2.0)
                lefttopy = int(y_center-height/2.0)
                roi = img[lefttopy:lefttopy + height, lefttopx:lefttopx + width]
                box = [lefttopx, lefttopy, lefttopx+width, lefttopy+height, aa[0]]
                bboxes.append(box)
        # print(bboxes)
        new_image = draw_bbox_new(img, bboxes, classes, filename)
        # new_image = cv2.cvtColor(new_image, cv2.COLOR_BGR2RGB)
        # plt.imshow(new_image)
        # plt.show()
        cv2.imwrite(os.path.join(save_path, filename), new_image)

        cv2.waitKey(0)

