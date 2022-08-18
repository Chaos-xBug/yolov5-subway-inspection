# 20220809
# python 3.7
# utf-8
# 建立分类数据集

import os
import shutil
import random


def copy_file(ori_addr, target_addr):
    shutil.copyfile(ori_addr, target_addr)


def set_dir(addr):
    for i in li:
        os.path.makedirs(addr + i + '/images/')
        os.path.makedirs(addr + i + '/labels/')
    

if __name__ == '__main__':
    img_suffix = '.jpg'
    root_path = '/'
    ori_img_path = 'images/'
    ori_txt_path = 'labels/'

    trainval_percent = 0.9  # 训练和验证集所占比例，剩下的0.1就是测试集的比例
    train_percent = 0.8  # 训练集所占比例，可自己进行调整

    li = ['test', 'train', 'val']

    set_dir(root_path, li)

    total_txt = os.listdir(root_path + ori_txt_path)
    num = len(total_txt)
    list = range(num)
    tv = int(num * trainval_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(list, tv)
    train = random.sample(trainval, tr)

    for i in list:
        name = total_txt[i][:-4]
        if i in trainval:
            pass
            if i in train:
                copy_file(root_path + ori_img_path + name + img_suffix, 'train/images/' + name + img_suffix)
                copy_file(root_path + ori_txt_path + name + '.txt', 'train/labels/' + name + '.txt')
            else:
                copy_file(root_path + ori_img_path + name + img_suffix, 'val/images/' + name + img_suffix)
                copy_file(root_path + ori_txt_path + name + '.txt', 'val/labels/' + name + '.txt')
        else:
            copy_file(root_path + ori_img_path + name + img_suffix, 'test/images/' + name + img_suffix)
            copy_file(root_path + ori_txt_path + name + '.txt', 'test/labels/' + name + '.txt')
