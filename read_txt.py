'''
遍历test.txt中图片文件路径，并保存在新的文件夹中。
'''

import cv2
import os

txt_path = '/home/wjd/code/mmdetection-master/data/VOCdevkit/VOC2007/test.txt'
save_path = '/home/wjd/code/mmdetection-master/data/coco/test2017/VOC2007/JPEGImages'


with open(txt_path, "r+") as f:
    for line in f:
        # aa = line.readline()
        aa = str(line).replace("\n", "")
        print(aa)
        img = cv2.imread(aa)
        # print(img.shape)
        # img_filename = str(n) + '.jpg'
        img_filename = os.path.split(aa)[1]
        print(img_filename)
        path2 = os.path.join(save_path, img_filename)
        cv2.imwrite(path2, img)

