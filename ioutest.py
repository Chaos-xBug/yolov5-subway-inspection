import numpy as np
import os

def compute_iou(box1, box2, wh=True):
    """
    compute the iou of two boxes.
    Args:
        box1, box2: [xmin, ymin, xmax, ymax] (wh=False) or [xcenter, ycenter, w, h] (wh=True)
        wh: the format of coordinate.
    Return:
        iou: iou of box1 and box2.
    """
    if wh == False:
        xmin1, ymin1, xmax1, ymax1 = box1
        xmin2, ymin2, xmax2, ymax2 = box2
    else:
        xmin1, ymin1 = int(box1[0] - box1[2] / 2.0), int(box1[1] - box1[3] / 2.0)
        xmax1, ymax1 = int(box1[0] + box1[2] / 2.0), int(box1[1] + box1[3] / 2.0)
        xmin2, ymin2 = int(box2[0] - box2[2] / 2.0), int(box2[1] - box2[3] / 2.0)
        xmax2, ymax2 = int(box2[0] + box2[2] / 2.0), int(box2[1] + box2[3] / 2.0)

    ## 获取矩形框交集对应的左上角和右下角的坐标（intersection）
    xx1 = np.max([xmin1, xmin2])
    yy1 = np.max([ymin1, ymin2])
    xx2 = np.min([xmax1, xmax2])
    yy2 = np.min([ymax1, ymax2])

    ## 计算两个矩形框面积
    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    inter_area = (np.max([0, xx2 - xx1])) * (np.max([0, yy2 - yy1])) # 计算交集面积
    iou = inter_area / (area1 + area2 - inter_area + 1e-6) #计算交并比

    return iou


if __name__ == "__main__":
    # 依次拿检测到的gt_boxes和真值ground truth中gt_box匹配，计算两者IOU，假设det_box为a，gt_box=b
    path_base = 'data\\Line3\\'
    w=h=5120
    filename = '1.bmp'
    # for filename in os.listdir(os.path.join(path_base, "Defaut")):
    path_label_o = 'data\\Line3\\StandardDefautTree\\labels\\'
    path_label_check = 'data\\Line3\\DefautTestResult\\labels\\'
    filename_txt = filename.replace('.bmp', '.txt')
    print(filename_txt)
    # 将ground_truth中全部n1个框坐标存入gt_boxes
    with open(os.path.join(path_label_o, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
        matrix = np.zeros((200, 4))
        n1 = 0
        for line in f:
            aa = line.split(" ")
            a = str(aa[0])
            # print(label)
            x_center = w * float(aa[1])  # aa[1]左上点的x坐标
            y_center = h * float(aa[2])  # aa[2]左上点的y坐标
            width = int(w * float(aa[3]))  # aa[3]图片width
            height = int(h * float(aa[4]))  # aa[4]图片height
            b = [x_center, y_center, width, height]
            # print(b)
            matrix[n1, :] = b
            n1=n1+1
        gt_boxes = matrix[0:n1, :]
    # 将detected的全部n2个框坐标存入det_boxes
    with open(os.path.join(path_label_check, filename_txt), "r+", encoding="utf-8", errors="ignore") as f:
        matrix = np.zeros((200, 4))
        n2 = 0
        for line in f:
            aa = line.split(" ")
            a = str(aa[0])
            # print(label)
            x_center = w * float(aa[1])  # aa[1]左上点的x坐标
            y_center = h * float(aa[2])  # aa[2]左上点的y坐标
            width = int(w * float(aa[3]))  # aa[3]图片width
            height = int(h * float(aa[4]))  # aa[4]图片height
            b = [x_center, y_center, width, height]
            # print(b)
            matrix[n2, :] = b
            n2=n2+1
        det_boxes = matrix[0:n2, :]

    # IOU计算如下，一共n1*n2个,第m行是第m个dt_box和n1个gt_box的IOU
    print(n2)
    print(n1)
    # print(n1.type())
    IOU = np.zeros((n2, n1))
    for i in range(n2):
        for j in range(n1):
            IOU[i,j] = compute_iou(det_boxes[i,:], gt_boxes[j,:])
    # print('IOU:', IOU)
    print('IOU_shape:', IOU.shape)

    # 找出每个det_box和三个gt_box IOU最值
    iou_max = np.max(IOU, axis=1)
    print('iou_max:', iou_max)
    print(iou_max.shape)
    # iou_max = [0.8 0.3 0.9 0.9]
    # 对IOU进行过滤小于0.5淘汰
    iou_max_index = iou_max > 0.5
    print('iou_max_index:', iou_max_index)  # iou_max_index=[ True  True False  True]
    new_det_boxes = det_boxes[iou_max_index]
    # [list([1, 2, 4, 5]) list([4, 5, 6, 7]) list([8, 9, 10, 11, 12])
    # 已经过滤掉det_box的第四个box
    # 这时候需要给每个det_box 匹配对应的gt_box,首先去除IOU中那个属于错误检测框的IOU,也就是IOU第三列
    gt_match_dt_index = np.argmax(IOU, axis=0)  # 返回给gt_box匹配IOU最大的对应det_box的索引
    print('gt_match_dt_index:', gt_match_dt_index)  # [1 0 2]
    # new_gt_boxes = gt_boxes[gt_match_dt_index]  #
    # print('new_gt_boxes:',new_gt_boxes)
    # [list([1, 2, 4, 5]) list([4, 5, 6, 7]) list([8, 9, 10, 11, 12])
    # 完成检测框和真值框的一一匹配

