# -*- coding: utf-8 -*- 
# @Time : 2021/4/23 下午3:16 
# @Author : coplin 
# @File : yolo2voc.py
# @desc:

import cv2
import os

xml_head = '''<annotation>
    <folder>VOC2007</folder>
    <!--文件名-->
    <filename>{}</filename>
    <source>
        <database>The VOC2007 Database</database>
        <annotation>PASCAL VOC2007</annotation>
        <images>flickr</images>
        <flickrid>325991873</flickrid>
    </source>
    <owner>
        <flickrid>null</flickrid>
        <name>null</name>
    </owner>    
    <size>
        <width>{}</width>
        <height>{}</height>
        <depth>{}</depth>
    </size>
    <segmented>0</segmented>
    '''
xml_obj = '''
    <object>        
        <name>{}</name>
        <pose>Rear</pose>
        <!--是否被裁减，0表示完整，1表示不完整-->
        <truncated>0</truncated>
        <!--是否容易识别，0表示容易，1表示困难-->
        <difficult>0</difficult>
        <!--bounding box的四个坐标-->
        <bndbox>
            <xmin>{}</xmin>
            <ymin>{}</ymin>
            <xmax>{}</xmax>
            <ymax>{}</ymax>
        </bndbox>
    </object>
    '''
xml_end = '''
</annotation>'''

# label for datasets
labels = ['Bolt', 'KeyholeCap', 'SplitPin', 'RubberPlug', 'Valve', 'Oil', 'TemperaturePaste', 'Nameplate', 'Bandage', 'Hoop',
        'UnionNut', 'PipeClamp', 'CrossBolt', 'BellowsHoop', 'BrakeWireJoint', 'CableJoint', 'GroundTerminal', 'Fastening', 'WheelTread', 'BrakeShoe',
        'Grille', 'AirOutlet', 'RubberPile', 'Sewing', 'Box', 'RectConnector', 'SteelWire', 'PressureTestPort', 'CircleConnect',
        'Wheellabel', 'MountingBolt', 'Scupper', 'PullTab', 'PlasticPlug', 'DrainCock', 'Warning', 'BoxCoverHinge', 'noBolt']

cnt = 0
txt_path = os.path.join(r'C:/Users/Lenovo/PycharmProjects/yolov5-wjd/voc/labels/')  # txt(yolo)文件目录
image_path = os.path.join(r'C:/Users/Lenovo/PycharmProjects/yolov5-wjd/voc/images/')  # 图片文件目录
path = os.path.join('C:/Users/Lenovo/PycharmProjects/yolov5-wjd/voc/Annotations/')  # xml(voc)文件目录

for (root, dirname, files) in os.walk(txt_path):  # 遍历图片文件夹
    for ft in files:
        print(ft)
        fjpg = ft.replace('txt', 'bmp')  # ft是图片名字+扩展名，将jpg和txt替换
        fxml = ft.replace('txt', 'xml')
        xml_path = path + fxml
        obj = ''
        img = cv2.imread(image_path + fjpg)
        img_h, img_w = img.shape[0], img.shape[1]
        head = xml_head.format(str(fxml), str(img_w), str(img_h), 3)

        with open(txt_path + ft, 'r') as f:  # 读取对应txt文件内容
            for line in f.readlines():
                yolo_datas = line.strip().split(' ')
                label = int(float(yolo_datas[0].strip()))
                center_x = round(float(str(yolo_datas[1]).strip()) * img_w)
                center_y = round(float(str(yolo_datas[2]).strip()) * img_h)
                bbox_width = round(float(str(yolo_datas[3]).strip()) * img_w)
                bbox_height = round(float(str(yolo_datas[4]).strip()) * img_h)

                xmin = str(int(center_x - bbox_width / 2))
                ymin = str(int(center_y - bbox_height / 2))
                xmax = str(int(center_x + bbox_width / 2))
                ymax = str(int(center_y + bbox_height / 2))

                obj += xml_obj.format(labels[label], xmin, ymin, xmax, ymax)
        with open(xml_path, 'w') as f_xml:
            f_xml.write(head + obj + xml_end)
        cnt += 1
        print(cnt)
