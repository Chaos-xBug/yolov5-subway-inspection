import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import copy

from markline.toolfunc import cal_pt_distance, letterbox_image

def detect_un_line(img):

    old_img = img
    old_img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst_img = cv2.blur(img, [5, 5])
    hsv_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)

    mask1 = cv2.inRange(hsv_img, np.array([0, 63, 33]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_img, np.array([163, 80, 50]), np.array([180, 255, 255]))
    mask = mask1 | mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel2)

    contours, hierarchy = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    num = len(contours)

    return old_img, close, contours, num, old_img1, dst_img, hsv_img, mask, open


def find_line(image):

    old_img, close, contours, num, old_img1, dst_img, hsv_img, mask, open = detect_un_line(image)

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


def UnionNut_State(image):

    num, angles, points, contours, old_img = find_line(image)

    if num == 0:
        state = -3
        # print("没有检测到有效的轮廓线")

    elif num == 1:
        state = 1
        # print("检测到一条有效标记线，没有松动")

    elif num == 2:

        # print("检测到两条有效标记线")

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
        print(min_Dis)
        if min_Dis > 0.1:
            state = 1
            # print("是两个距离比较远的线，没有松动")
        else:

            x1_angle = angles[0]
            y1_angle = angles[1]
            x2_angle = angles[2]
            y2_angle = angles[3]

            angle1 = math.atan2(y1_angle, x1_angle)
            angle1 = int(angle1 * 180 / math.pi)

            angle2 = math.atan2(y2_angle, x2_angle)
            angle2 = int(angle2 * 180 / math.pi)

            if angle1 * angle2 >= 0:
                included_angle = abs(angle1 - angle2)
            else:
                included_angle = abs(angle1) + abs(angle2)
                if included_angle > 180:
                    included_angle = 360 - included_angle

            # print("两条轮廓线之间的角度", included_angle)

            if included_angle <= 20 or included_angle >= 160:
                state = 1
                # print("同一个物体上的两条线没有松动")
            else:
                state = 0
                # print("同一个物体上的两条线松动了")

        # 这是一个比较简单的判别逻辑，检测两条标记线之间的相对距离，设定一个阈值（测试得到）
        # 如果距离大于这个阈值，那么就说明是两个不同的螺栓的标记线，这种情况是没有松动的
        # 如果距离小于这个阈值，就判别角度，（但应该再加上一个关于距离的阈值，用来区分刚好松动180度的情况，但目前没有实验数据，阈值还无法确定）

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
        # print(angle1)  # 轮廓0

        angle2 = math.atan2(y2_angle, x2_angle)
        angle2 = int(angle2 * 180 / math.pi)
        # print(angle2)  # 轮廓1

        angle3 = math.atan2(y3_angle, x3_angle)
        angle3 = int(angle3 * 180 / math.pi)
        # print(angle3)  # 轮廓2

        angless = []
        if angle1 * angle2 >= 0:
            included_angle1 = abs(angle1 - angle2)
        else:
            included_angle12 = abs(abs(angle1) - abs(angle2))
            included_angle11 = abs(angle1) + abs(angle2)
            included_angle1 = min(included_angle11, included_angle12)

        if included_angle1 > 180:
            included_angle1 = 360 - included_angle1
        if included_angle1 > 90:
            included_angle1 = 180 - included_angle1
        # 轮廓0 和 轮廓1 的夹角
        angless.append(included_angle1)

        if angle1 * angle3 >= 0:
            included_angle2 = abs(angle1 - angle3)
        else:
            included_angle21 = abs(abs(angle1) - abs(angle3))
            included_angle22 = abs(angle1) + abs(angle3)
            included_angle2 = min(included_angle21, included_angle22)
        if included_angle2 > 180:
            included_angle2 = 360 - included_angle2
        if included_angle2 > 90:
            included_angle2 = 180 - included_angle2
        # 轮廓0 和 轮廓2 的夹角
        angless.append(included_angle2)

        if angle2 * angle3 >= 0:
            included_angle3 = abs(angle2 - angle3)
        else:
            included_angle31 = abs(abs(angle2) - abs(angle3))
            included_angle32 = abs(angle2) + abs(angle2)
            included_angle3 = min(included_angle31, included_angle32)
        if included_angle3 > 180:
            included_angle3 = 360 - included_angle3
        if included_angle3 > 90:
            included_angle3 = 180 - included_angle3
        # 轮廓1 和 轮廓2 的夹角
        angless.append(included_angle3)

        # print("夹角分别为：", angless)

        # 距离
        dises = []

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
        dises.append(min_Dis1)  # contour0和contour1的距离

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
        dises.append(min_Dis2)  # contour0 和 contour2的距离

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
        dises.append(min_Dis3)  # contour1 和 contour2的距离

        h, w, c = old_img.shape
        a = [i/w for i in dises]
        # print("距离分别为：", a)

        if min(a) > 0.07:
            state = 1
            # print("是三个不同的线")
        else:

            min_Dis = min(dises)  # 求列表最小值
            min_idx = dises.index(min_Dis)  # 求最小值对应索引

            if angless[min_idx] < 20:
                state = 1
                # print("没有松动")
            else:
                state = 0
                # print("松动了")

        # 这里的逻辑是如果检测到三条轮廓线。分别求轮廓0和1之间的角度、距离  轮廓0和轮廓2之间的角度、距离  轮廓1和轮廓2之间的角度、距离
        # 找出距离比较小的两个轮廓，这两个轮廓应该是同一个螺栓上检测出来的两个轮廓，通过角度判别这个螺栓有没有发生松动
        # 依然是少了相对距离判断是否松动180度
        # 以及抗干扰很能力很差，因为如果图中确实就有三条螺栓的话也适用于这个规则，所以可能还要在对单个螺栓设定两个阈值

    # elif num == 4:
    #     print("检测到四条有效标记线")
    #
    #     x1_angle = angles[0]
    #     y1_angle = angles[1]
    #     x2_angle = angles[2]
    #     y2_angle = angles[3]
    #     x3_angle = angles[4]
    #     y3_angle = angles[5]
    #     x4_angle = angles[6]
    #     y4_angle = angles[7]
    #
    #
    #     angle1 = math.atan2(y1_angle, x1_angle)
    #     angle1 = int(angle1 * 180 / math.pi)
    #     print(angle1)  # 轮廓0
    #
    #     angle2 = math.atan2(y2_angle, x2_angle)
    #     angle2 = int(angle2 * 180 / math.pi)
    #     print(angle2)  # 轮廓1
    #
    #     angle3 = math.atan2(y3_angle, x3_angle)
    #     angle3 = int(angle3 * 180 / math.pi)
    #     print(angle3)  # 轮廓2
    #
    #     angle4 = math.atan2(y4_angle, x4_angle)
    #     angle4 = int(angle4 * 180 / math.pi)
    #     print(angle4)  # 轮廓3
    #
    #     angless = []
    #     angless.append(angle1)
    #     angless.append(angle2)
    #     angless.append(angle3)
    #     angless.append(angle4)
    #
    #     angle_diff = []
    #     i = 0
    #     while i < len(angless):
    #         for j in range((i+1), len(angless)):
    #             if angless[i] * angless[j] >= 0:
    #                 included_angle = abs(angless[i] - angless[j])
    #             else:
    #                 included_angle12 = abs(abs(angless[i]) - abs(angless[j]))
    #                 included_angle11 = abs(angless[i]) + abs(angless[j])
    #                 included_angle = min(included_angle11, included_angle12)
    #
    #             if included_angle > 180:
    #                 included_angle = 360 - included_angle
    #             if included_angle > 90:
    #                 included_angle = 180 - included_angle
    #
    #             angle_diff.append(included_angle)
    #             j += 1
    #         i += 1
    #
    #     # 这时候angle_diff中就 01 02 03 12 13 23 轮廓之间的角度差
    #
    #     print("角度差分别为：", angle_diff)
    #
    #     if max(angle_diff) < 20:
    #         state = 1
    #     else:
    #
    #         # 下面求距离，四条轮廓线的逻辑是， 找到距离最小的两对轮廓线， 对这两对轮廓线求角度
    #         dises = []
    #
    #         i = 0
    #         while i < num-1:
    #
    #             j = i + 1
    #             while j < num:
    #
    #                 min_dis = 1000
    #
    #                 for k in range(len(contours[i])):
    #                     pt = tuple(contours[i][k][0])
    #
    #
    #                     for m in range(len(contours[j])):
    #                         pt2 = tuple(contours[j][m][0])
    #                         distance = cal_pt_distance(pt, pt2)
    #                         if distance < min_dis:
    #                             min_dis = distance
    #                 j += 1
    #
    #                 dises.append(min_dis)
    #
    #             i += 1
    #         print("最小距离分别是：", dises)
    #
    #
    #         t = copy.deepcopy(dises)
    #         # 求m个最大的数值及其索引
    #         min_number = []
    #         min_index = []
    #         for _ in range(2):
    #             number = min(t)
    #             index = t.index(number)
    #             t[index] = 1000
    #             min_number.append(number)
    #             min_index.append(index)
    #
    #         a = min_index[0]
    #         b = min_index[1]
    #         if angle_diff[a] < 20 and angle_diff[b] < 20:
    #             state = 1
    #             print("没松动")
    #         else:
    #             state = 0
    #             print("松动了")

    else:
        # print("标记线检测有问题")
        state = -2

    return state, old_img


if __name__ == "__main__":

    root_path = "/Users/kerwinji/Desktop/tagline_detect/unionnut_result/511/wu"
    image_path = os.path.join(root_path, "UnionNut5_Pic_2022_05_12_153558_195.bmp")
    image = cv2.imread(image_path)

    old_img, close, contours1, num, old_img1, dst_img, hsv_img, mask, open = detect_un_line(image)

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
    state, old_img3 = UnionNut_State(image)

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



    # 这个就是检测到的点之间的相对距离比较难确定，（判断是否松动那里）