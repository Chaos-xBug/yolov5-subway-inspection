# -*- coding: utf-8 -*-
# 将检测结果按目标种类切分并保存
import os
import cv2

path_txt = r'E:\04-Data\02-robot\05-Xingqiao0708\Standard\completeresults\labels\\' # jpg图片和对应的生成结果的txt标注文件，放在一起
path_image = r"E:\04-Data\02-robot\05-Xingqiao0708\Standard\complete\\"                  # 测试图片路径
save_path = r'E:\04-Data\02-robot\06-Xingqiao0722\cutimages\144554\\'  # 裁剪出来的小图保存的根目录
# w = 1920
# h = 1080
classes = ['Bolt', 'KeyholeCap', 'SplitPin', 'RubberPlug', 'Valve', 'Oil', 'TemperaturePaste', 'Nameplate', 'Bandage', 'Hoop',
        'UnionNut', 'PipeClamp', 'CrossBolt', 'BellowsHoop', 'BrakeWireJoint', 'CableJoint', 'GroundTerminal', 'Fastening', 'WheelTread', 'BrakeShoe',
        'Grille', 'AirOutlet', 'RubberPile', 'Sewing', 'Box', 'RectConnector', 'SteelWire', 'PressureTestPort', 'CircleConnect',
        'Wheellabel', 'MountingBolt', 'Scupper', 'PullTab', 'PlasticPlug', 'DrainCock', 'Warning', 'BoxCoverHinge', 'noBolt']
n = 38
for i in range(n):
    isExists = os.path.exists(save_path+classes[i])
    if not isExists:						#判断如果文件不存在,则创建
        os.makedirs(save_path+classes[i])
        print("%s 目录创建成功"%i)
    else:
        print("%s 目录已经存在"%i)
        continue			#如果文件不存在,则继续上述操作,直到循环结束

#
# for(root, files, filename) in os.walk(path_image):    # 遍历文件名
#     print(filename)
#     for filename in filename:
#         print('filename_img:', filename)
#         path1 = os.path.join(path_image, str(filename))
#         # print(path1)
#         img = cv2.imread(path1)
#         h = img.shape[0]
#         w = img.shape[1]
#         # img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)        # resize 图像大小，否则roi区域可能会报错
#         filename_txt = filename.replace('.jpg', '.txt')
#         # print('filename_txt:', filename_txt)
#         n = 1
#         with open(os.path.join(path_txt, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
#             for line in f:
#                 aa = line.split(" ")
#                 a = str(aa[0])
#                 label = classes[int(a)]  # 裁剪出来的小图保存的根目录
#                 # print(label)
#                 x_center = w * float(aa[1])       # aa[1]左上点的x坐标
#                 y_center = h * float(aa[2])       # aa[2]左上点的y坐标
#                 width = int(w*float(aa[3]))       # aa[3]图片width
#                 height = int(h*float(aa[4]))      # aa[4]图片height
#                 lefttopx = int(x_center-width/2.0)
#                 lefttopy = int(y_center-height/2.0)
#                 # roi = img[lefttopy:lefttopy + height, lefttopx:lefttopx + width]
#                 # print('roi_size_s:', roi.shape)
#                 lens = 20
#                 roi = img[abs(lefttopy-lens):lefttopy+height+lens, abs(lefttopx-lens):lefttopx+width+lens]   # [左上y:右下y,左上x:右下x] (y1:y2,x1:x2)需要调参，否则裁剪出来的小图可能不太好
#                 # print('roi_size:', roi.shape)                       # 如果不resize图片统一大小，可能会得到有的roi为[]导致报错
#                 # if roi.shape[1] == 0:
#                 #     print(lefttopx-10)
#                 filename_last = label+str(n)+'_'+filename    # 裁剪出来的小图文件名
#                 # print(filename_last)
#                 path2 = os.path.join(save_path, label)           # 需要在path3路径下创建一个roi文件夹
#                 # print('path2:', path2)                    # 裁剪小图的保存位置
#                 cv2.imwrite(os.path.join(path2, filename_last), roi)
#                 n = n+1
