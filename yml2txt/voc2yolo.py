# -*- coding: utf-8 -*-
# @Time : 2021/4/20 下午2:25
# @Author : coplin
# @File : voc2yolo.py
# @desc:

import xml.etree.ElementTree as ET
import os

sets = ['train', 'val', 'test']

# 个状
# classes = ["Bolt", "KeyholeCap", "SplitPin", "RubberPlug", "Valve", "Oil", "TemperaturePaste", "Nameplate", "Bandage",
#            "Hoop", "UnionNut", "PipeClamp", "CrossBolt", "Scupper", "PullTab", "PlasticPlug", "DrainCock", "Warning",
#            "Connector", "BoxCoverHinge", "MountingBolt"]  # 类别
# rootpath = '/home/crrc/workspaces/data/data-voc-0425/'


# 线状
# classes = ['GroundWire', 'BrakeWire', 'CorrugatedPipe', 'Wire', 'AirTube']
# rootpath = '/home/crrc/workspaces/data/line-data-voc/'

classes = ['Bolt', 'KeyholeCap', 'SplitPin', 'RubberPlug', 'Valve', 'Oil', 'TemperaturePaste', 'Nameplate', 'Bandage', 'Hoop',
        'UnionNut', 'PipeClamp', 'CrossBolt', 'BellowsHoop', 'BrakeWireJoint', 'CableJoint', 'GroundTerminal', 'Fastening', 'WheelTread', 'BrakeShoe',
        'Grille', 'AirOutlet', 'RubberPile', 'Sewing', 'Box', 'RectConnector', 'SteelWire', 'PressureTestPort', 'CircleConnect',
        'Wheellabel', 'MountingBolt', 'Scupper', 'PullTab', 'PlasticPlug', 'DrainCock', 'Warning', 'BoxCoverHinge', 'noBolt']

rootpath = '../../voc/train/'


def convert(size, box):
    dw = 1. / (size[0])
    dh = 1. / (size[1])
    x = (box[0] + box[1]) / 2.0 - 1
    y = (box[2] + box[3]) / 2.0 - 1
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return x, y, w, h


def convert_annotation(image_id):
    in_file = open(rootpath + 'Annotations/%s.xml' % (image_id), encoding='UTF-8')
    out_file = open(rootpath + 'labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        difficult = obj.find('difficult').text

        cls = obj.find('name').text
        if cls not in classes or int(difficult) == 1:
            continue
        cls_id = classes.index(cls)
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        b1, b2, b3, b4 = b
        # 标注越界修正
        if b2 > w:
            b2 = w
        if b4 > h:
            b4 = h
        b = (b1, b2, b3, b4)
        bb = convert((w, h), b)
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


for image_set in sets:
    if not os.path.exists(rootpath + 'labels/'):
        os.makedirs(rootpath + 'labels/')

    image_ids = open(rootpath + 'ImageSets/Main/%s.txt' % (image_set)).read().strip().split()
    list_file = open(rootpath + '%s.txt' % (image_set), 'w')

    for image_id in image_ids:
        list_file.write(rootpath + 'JPEGImages/%s.jpg\n' % (image_id))
        convert_annotation(image_id)
    list_file.close()
