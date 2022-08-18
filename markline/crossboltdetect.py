import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os

from markline.toolfunc import cal_pt_distance, letterbox_image

def detect_cb_line(img):

    old_img = img
    old_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst_img = cv2.blur(img, [5, 5])
    hsv_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv_img, np.array([0, 63, 33]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_img, np.array([163, 80, 50]), np.array([180, 255, 255]))
    mask = mask1 | mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel2)

    contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num = len(contours)
    # print("检测到的轮廓数：", num)

    return old_img, close, contours, num, old_img1, dst_img, hsv_img, mask, open


def find_line(image):

    old_img, close, contours, num, old_img1, dst_img, hsv_img, mask, open = detect_cb_line(image)

    w1, h1, c1 = old_img.shape
    area1 = w1 * h1

    valid_contours = []
    points = []
    angles = []

    i = 0
    while i < num:

        (x, y), (w, h), angle = cv2.minAreaRect(contours[i])
        wh_ratio = float(w) / h
        hw_ratio = h / float(w)
        area_con = cv2.contourArea(contours[i])
        s = area_con / area1
        if s >= 0.005 and (wh_ratio > 1.1 or hw_ratio > 1.1):
            valid_contours.append(contours[i])
            i += 1
        else:
            i += 1

    num1 = len(valid_contours)

    j = 0
    while j < num1:

        line = cv2.fitLine(valid_contours[j], cv2.DIST_L2, 0, 0.01, 0.01)

        angles.append(line[0][0])
        angles.append(line[1][0])
        points.append(line[2][0])
        points.append(line[3][0])

        j += 1

    return num1, angles, points, valid_contours, old_img


def CrossBolt_State(image):

    num, angles, points, contours, old_img = find_line(image)

    if num == 0:
        state = -3
        # print("没有检测到有效的轮廓线")

    elif num == 1:
        state = 1
        # print("检测到一条有效标记线，没有松动")

    elif num == 2:

        # print("检测到两条有效标记线")

        x1_angle = angles[0]
        y1_angle = angles[1]
        x2_angle = angles[2]
        y2_angle = angles[3]

        angle1 = math.atan2(y1_angle, x1_angle)
        angle1 = int(angle1 * 180/math.pi)

        angle2 = math.atan2(y2_angle, x2_angle)
        angle2 = int(angle2 * 180/math.pi)

        if angle1 * angle2 >= 0:
            included_angle = abs(angle1-angle2)
        else:
            included_angle = abs(angle1) + abs(angle2)
            if included_angle > 180:
                included_angle = 360 - included_angle

        if included_angle <= 20 or included_angle >= 160:
            anglestate = 1
        else:
            anglestate = 0

        # 计轮廓间最近的两点之间的距离
        min_Dis = 1000
        for i in range(len(contours[0])):
            pt = tuple(contours[0][i][0])
            min_dis = 1000
            for j in range(0, len(contours[1])):
                pt2 = tuple(contours[1][j][0])
                distance = cal_pt_distance(pt, pt2)
                if distance < min_dis:
                    min_dis = distance

            if min_dis < min_Dis:
                min_Dis = min_dis

        h, w, c = old_img.shape
        min_Dis = min_Dis / w

        if min_Dis <= 0.2:
            pointstate = 1
        else:
            pointstate = 0

        if anglestate == 1 and pointstate == 1:
            state = 1
            # print("没有松动", "角度差：", included_angle, "相对距离差：", min_Dis)
        else:
            state = 0
            # print("松动了", "角度差：", included_angle, "相对距离差：", min_Dis)

    elif num == 3:

        # print("检测到三条有效标记线")

        x1_angle = angles[0]
        y1_angle = angles[1]
        x2_angle = angles[2]
        y2_angle = angles[3]
        x3_angle = angles[4]
        y3_angle = angles[5]

        angle1 = math.atan2(y1_angle, x1_angle)
        angle1 = int(angle1 * 180 / math.pi)

        angle2 = math.atan2(y2_angle, x2_angle)
        angle2 = int(angle2 * 180 / math.pi)

        angle3 = math.atan2(y3_angle, x3_angle)
        angle3 = int(angle3 * 180 / math.pi)

        if angle1 * angle2 >= 0:
            included_angle1 = abs(angle1 - angle2)
        else:
            included_angle12 = abs(abs(angle1) - abs(angle2))
            included_angle11 = abs(angle1) + abs(angle2)
            included_angle1 = min(included_angle11, included_angle12)

            if included_angle1 > 180:
                included_angle1 = 360 - included_angle1

        if angle1 * angle3 >= 0:
            included_angle2 = abs(angle1 - angle3)
        else:
            included_angle21 = abs(abs(angle1) - abs(angle3))
            included_angle22 = abs(angle1) + abs(angle3)
            included_angle2 = min(included_angle21, included_angle22)
            if included_angle2 > 180:
                included_angle2 = 360 - included_angle2

        if angle2 * angle3 >= 0:
            included_angle3 = abs(angle2 - angle3)
        else:
            included_angle31 = abs(abs(angle2) - abs(angle3))
            included_angle32 = abs(angle2) + abs(angle2)
            included_angle3 = min(included_angle31, included_angle32)
            if included_angle3 > 180:
                included_angle3 = 360 - included_angle3

        included_angle = min(included_angle1, included_angle2, included_angle3)

        if included_angle <= 20 or included_angle >= 160:
            anglestate = 1
        else:
            anglestate = 0

        # 简单repeat一下num=2的逻辑
        min_Dis1 = 1000
        for i in range(len(contours[0])):
            pt = tuple(contours[0][i][0])
            min_dis1 = 1000
            for j in range(0, len(contours[1])):
                pt2 = tuple(contours[1][j][0])
                distance = cal_pt_distance(pt, pt2)
                if distance < min_dis1:
                    min_dis1 = distance

            if min_dis1 < min_Dis1:
                min_Dis1 = min_dis1

        min_Dis2 = 1000
        for i in range(len(contours[0])):
            pt = tuple(contours[0][i][0])
            min_dis2 = 1000
            for j in range(0, len(contours[2])):
                pt2 = tuple(contours[2][j][0])
                distance = cal_pt_distance(pt, pt2)
                if distance < min_dis2:
                    min_dis2 = distance

            if min_dis2 < min_Dis2:
                min_Dis2 = min_dis2

        min_Dis3 = 1000
        for i in range(len(contours[2])):
            pt = tuple(contours[2][i][0])
            min_dis3 = 1000
            for j in range(0, len(contours[1])):
                pt2 = tuple(contours[1][j][0])
                distance = cal_pt_distance(pt, pt2)
                if distance < min_dis3:
                    min_dis3 = distance

            if min_dis3 < min_Dis3:
                min_Dis3 = min_dis3

        min_Dis = min(min_Dis1, min_Dis2, min_Dis3)

        h, w, c = old_img.shape
        min_Dis = min_Dis / w

        if min_Dis <= 0.2:
            pointstate = 1
        else:
            pointstate = 0

        if anglestate == 1 and pointstate == 1:
            state = 1
            # print("没有松动", "角度差：", included_angle, "相对距离差：", min_Dis)
        else:
            state = 0
            # print("松动了", "角度差：", included_angle, "相对距离差：", min_Dis)

    else:
        # print("标记线检测有问题")
        state = -2

    return state, old_img


if __name__ == "__main__":

    root_path = "/Users/kerwinji/Desktop/tagline_detect/crossbolt_result/513/wu"
    image_path = os.path.join(root_path, "CrossBolt86_Pic_2022_05_13_152845_81.bmp")
    image = cv2.imread(image_path)

    old_img, close, contours1, num, old_img1, dst_img, hsv_img, mask, open = detect_cb_line(image)

    w1, h1, c1 = old_img.shape
    area1 = w1 * h1

    for c in range(len(contours1)):
        area_con = cv2.contourArea(contours1[c])
        print("面积：", area_con / area1)
        (x, y), (w, h), angle = cv2.minAreaRect(contours1[c])
        wh_ratio = float(w) / h
        hw_ratio = h / float(w)
        print("宽高比", wh_ratio)
        print("高宽比", hw_ratio)

    num1, angles, points, contours, old_img2 = find_line(image)
    state, old_img3 = CrossBolt_State(image)

    fig, axes = plt.subplots(nrows=3, ncols=2)

    axes[0, 0].imshow(old_img1)
    axes[0, 0].set_title("image")

    axes[0, 1].imshow(dst_img)
    axes[0, 1].set_title("blur")

    axes[1, 0].imshow(hsv_img)
    axes[1, 0].set_title("hsv")

    axes[1, 1].imshow(mask)
    axes[1, 1].set_title("mask")

    axes[2, 0].imshow(open)
    axes[2, 0].set_title("open")

    axes[2, 1].imshow(close)
    axes[2, 1].set_title("close")
    plt.show()

    # 画轮廓
    cv2.drawContours(old_img, contours, -1, (0, 0, 255), 1)

    # 画出最小外接矩形
    for c in range(len(contours)):
        rect = cv2.minAreaRect(contours[c])
        box = cv2.boxPoints(rect)  # 返回矩形四个角点坐标
        box = np.int0(box)  # 获得矩形角点坐标(整数)
        cv2.drawContours(old_img, [box], -1, (0, 220, 0,), 2)
    # 还有利用最小外接矩形的角度来判别松动的可能行
    # (x, y), (w, h), angle = = cv2.minAreaRect(contours[i])

    # 画轮廓的拟合线
    j = 0
    if num1 > 0:
        while j < num1:
            M = cv2.moments(contours[j])
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            [vx, vy, x, y] = cv2.fitLine(contours[j], cv2.DIST_L2, 0, 0.01, 0.01)
            k = vy / vx
            b = y - k * x
            x2 = cx - 200
            y2 = int(k * x2 + b)
            x3 = cx + 200
            y3 = int(k * x3 + b)
            cv2.line(old_img, (x2, y2), (x3, y3), (0, 255, 0), 1)
            j += 1
    else:
        pass

    cv2.imshow("1", old_img)
    cv2.waitKey(0)
