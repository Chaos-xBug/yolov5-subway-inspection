# -*- coding: utf-8 -*-
# @Time : 2021/4/21 下午4:31
# @Author : coplin
# @File : crrcdetect.py
# @desc:
import argparse
import time
import matplotlib.pyplot as plt
from pathlib import Path
import os
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

def plotrectangle(boxes,image):
    for i in range(boxes.shape[0]):
        l = boxes[i, 4]
        m = boxes[i, 5]
        cv2.rectangle(image, (int(boxes[i, 2] - l / 2), int(boxes[i, 3] - m / 2)),
                      (int(boxes[i, 2] + l / 2), int(boxes[i, 3] + m / 2)), (0, 0, 255), 10)
    # # 画图检查
    # plt.imshow(image), plt.title('image with boxes'), plt.show()
    # cv2.imshow('image with boxes',image)
    # cv2.waitKey(0)
    # cv2.imwrite('img_check.jpg', image)
    return image

def normalize(det, w, h):
    matrix = np.zeros((np.shape(det)[0], 6))
    matrix[:, 1] = det[:, 0]
    matrix[:, 2]= w * (det[:, 1])  # aa[1]左上点的x坐标 x_center
    matrix[:, 3]= h * (det[:, 2])  # aa[2]左上点的y坐标 y_center
    matrix[:, 4]= (w * (det[:, 3]))  # aa[3]图片width
    matrix[:, 5] = (h * (det[:, 4]))  # aa[4]图片height
    # [编号，label, X, Y, W, H]
    return matrix

def initialze(weights):

    imgsz = 1088
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

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            lines = np.zeros([len(det), 5])
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Write results
                count = 0
                for *xyxy, conf, cls in reversed(det): # 坐标 置信度 label代号
                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    lines[count,:] = [int(cls), *xywh]  # label format
                    count += 1
    return normalize(lines, 5120, 5120)

if __name__ == '__main__':

    output=r'E:\04-Data\02-robot\04-Xingqiao0512\Splitpin\Result'
    project=r'runs/detect' #'save results to project/name'
    name=r'expSplitpin' #'save results to project/name'

    basepath = 'E:\\04-Data\\02-robot\\04-Xingqiao0512\\3-CollectedData\\'
    weights = r'C:/Users/Lenovo/PycharmProjects/yolov5-wjd/runs/train/best.pt'  # yolo检测权重

    check_requirements(exclude=('pycocotools', 'thop'))

    save_txt = False
    # Directories
    save_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    device, model, half, imgsz, stride = initialze(weights)
    for filename in os.listdir(os.path.join(basepath, "Newget")):
        source = os.path.join(basepath, r"Newget\\", filename)
        with torch.no_grad():
            det = detect(source, save_dir, device, model, half, imgsz, stride)

            im1 = cv2.imread(source)
            # plt.imshow(im1), plt.title('image'), plt.show()
            im2 = plotrectangle(det, im1)
            cv2.imwrite(filename, im2)