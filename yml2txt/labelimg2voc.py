# -*- coding: utf-8 -*- 
# @Time : 2021/4/20 下午2:14 
# @Author : coplin 
# @File : labelimg2voc.py
# @desc: 格式转换成VOC格式，并按比例切分

import os
import random

rootpath = '../../voc/train/'
trainval_percent = 0.9  # 训练和验证集所占比例，剩下的0.1就是测试集的比例
train_percent = 0.8  # 训练集所占比例，可自己进行调整
txtfilepath = rootpath + 'labels'
txtsavepath = rootpath + 'ImageSets'
total_txt = os.listdir(txtfilepath)

num = len(total_txt)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)

ftrainval = open(rootpath + 'ImageSets/Main/trainval.txt', 'w')
ftest = open(rootpath + 'ImageSets/Main/test.txt', 'w')
ftrain = open(rootpath + 'ImageSets/Main/train.txt', 'w')
fval = open(rootpath + 'ImageSets/Main/val.txt', 'w')

for i in list:
    name = total_txt[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)

ftrainval.close()
ftrain.close()
fval.close()
ftest.close()

# added 05172022 生成路径文件
sets = ['train', 'val', 'test']
for image_set in sets:

    image_ids = open(rootpath + 'ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open(rootpath + '%s.txt' % (image_set), 'w')

    for image_id in image_ids:
        list_file.write(rootpath + 'JPEGImages/%s.jpg\n' % (image_id))
    list_file.close()