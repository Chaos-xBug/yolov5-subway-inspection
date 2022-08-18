# -*- coding: utf-8 -*-
# @Time : 2021/4/21 下午4:31
# @Author : coplin
# @File : crrcdetect.py
# @desc:
import argparse
import time
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def initialze(weights):

    imgsz = 224
    # Initialize
    set_logging()
    device = select_device('cpu')  # 0 1 2
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    if half:
        model.half()  # to FP16
    return device, model, half, imgsz, stride

def detect(source, device, model, half, imgsz, stride):  # save_img=False
    res = 1
    im0 = []
    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names # [Head, Root]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inferencea
        pred = model(img, augment=False)[0]

        # Apply NMS
        conf_thres = 0.4
        iou_thres = 0.6  # IOU threshold for NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, agnostic=False)
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            res = 1
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                n0 = (det[:, -1] == 0).sum()  # detections of head
                n1 = (det[:, -1] == 1).sum()  # detections of root
                if n0 != 1:
                    res = -1
                    print('lost Head part of spilitpin')
                if n1 != 2:
                    res = -1
                    print('lost Root part of spilitpin')

                # Write results
                for *xyxy, conf, cls in reversed(det): # 坐标 置信度 label代号
                    # Add bbox to image
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=5)
            else:
                res = -1
                print('not detect any parts of splitpin')
            # cv2.imshow(str(p), im0)
            # cv2.waitKey(0)  #

    return res, im0

if __name__ == '__main__':

    basepath = r'E:\04-Data\02-robot\04-Xingqiao0512'
    weights=r'C:\Users\Lenovo\PycharmProjects\yolov5-wjd\runs\train\expSplitpin\last.pt'
    # source=r'E:\04-Data\02-robot\04-Xingqiao0512\Splitpin\SplitPin7_Pic_2022_05_13_113249_221.bmp'
    output=r'E:\04-Data\02-robot\04-Xingqiao0512\Splitpin\Result'
    project=r'runs/detect' #'save results to project/name'
    name=r'expSplitpin' #'save results to project/name'

    check_requirements(exclude=('pycocotools', 'thop'))

    save_txt = False
    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    device, model, half, imgsz, stride = initialze(weights)
    for filename in os.listdir(os.path.join(basepath, "Test")):

        # source = os.path.join(basepath, r"Test\\", filename)
        source = r'E:\04-Data\02-robot\04-Xingqiao0512\3-CollectedData\SplitpinParts\Pic_2022_05_12_145412_2.jpg'
        with torch.no_grad():
            res, img = detect(source, device, model, half, imgsz, stride)
            print('res:', res)