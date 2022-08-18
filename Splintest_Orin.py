import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import math
import markline.predict as tq

def detect_line(img, color = 'red'):
    # 进行中值滤波
    dst_img = cv2.medianBlur(img, 3)
    # plt.imshow(dst_img), plt.title('DST_Image'), plt.show()

    hsv_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)
    # plt.imshow(hsv_img), plt.title('HSV_Image'), plt.show()
    if color == 'red':
        mask1 = cv2.inRange(hsv_img, np.array([0, 80, 50]), np.array([12, 255, 255]))
        mask2 = cv2.inRange(hsv_img, np.array([175, 80, 50]), np.array([180, 255, 255]))
        mask = mask1 | mask2
        # plt.imshow(mask, 'gray'), plt.title('Mask'), plt.show()
    elif color == 'blue':
        mask = cv2.inRange(hsv_img, np.array([65, 30, 10]), np.array([95, 255, 255]))

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    open =cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    # plt.imshow(open), plt.title('Open'), plt.show()
    return open

def cal_distance(p1,p2):
    return math.sqrt(math.pow((p2[0]-p1[0]),2)+math.pow((p2[1]-p1[1]),2))

def draw_rbbox(img, points):
    # img = cv2.drawContours(img, [points], 0, (255, 0, 0), 1)
    rect = cv2.minAreaRect(points)
    print(rect)
    points_rect = cv2.boxPoints(rect)
    box = np.int0(points_rect)
    img = cv2.drawContours(img, [box], 0, (0, 0, 255), 1)
    return img

basepath = r'E:\\04-Data\\02-robot\\04-Xingqiao0512\\KeyHoleCap\\KeyholeCap\\'

for filename in os.listdir(basepath):
    # filename = basepath + 'KeyholeCap41_Pic_2022_05_13_150620_3.bmp'
    # filename = basepath + 'KeyholeCap39_Pic_2022_05_13_154747_155.bmp'
    # KeyholeCap39_Pic_2022_05_13_154747_155.bmp
    print(filename)
    img = cv2.imread(os.path.join(basepath,filename))
    # print(img.shape())
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    # plt.imshow(img), plt.title('Image'), plt.show()
    h, w = img_gray.shape
    if h/w >1.5 or h/w < 2/3:
        print('not value')
        continue
    '''
    Hough圆检测
    '''
    # 进行中值滤波
    dst_img = cv2.medianBlur(img_gray, 7)

    # 霍夫圆检测
    circle = cv2.HoughCircles(dst_img, cv2.HOUGH_GRADIENT, 1, 50,
                             param1=100, param2=100, minRadius=0, maxRadius=300)
    if circle is None:
        print('did not find circle')
        continue
    # 将检测结果绘制在图像上
    img11 = img.copy()
    for i in circle[0, :]:  # 遍历矩阵的每一行的数据
        # 绘制圆形
        cv2.circle(img11, (int(i[0]), int(i[1])), int(i[2]), (255, 0, 0), 10)
        # 绘制圆心
        cv2.circle(img11, (int(i[0]), int(i[1])), 3, (255, 0, 0), -1)

    # 显示图像
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
    axes[0].imshow(img[:, :, ::-1])
    axes[0].set_title("原图")
    axes[1].imshow(img11[:, :, ::-1])
    axes[1].set_title("霍夫圆检测后的图像")
    plt.show()

    '''
    提取红色蓝色
    '''
    # 提取检测圆区域
    x = circle[0, :][0][0]
    y = circle[0, :][0][1]
    r = circle[0, :][0][2]
    n=50
    img_s = img[max(int(x-r+n),0):int(x+r-n), max(int(y-r+n),0):int(y+r-n), :]
    img_s1 = img_s.copy()
    plt.imshow(img_s), plt.title('img_s'), plt.show()

    # 利用HSV模型进行红蓝标记检测
    img_redline = detect_line(img_s,'red')
    # plt.imshow(img_redline), plt.title('img_line'), plt.show()
    img_blueline = detect_line(img_s, 'blue')
    # plt.imshow(img_blueline), plt.title('img_line'), plt.show()

    # 寻找轮廓
    contoursred, hierarchy = cv2.findContours(img_redline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contoursblue, hierarchy_ = cv2.findContours(img_blueline, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contoursredshow = [x + [[xy[1].min(), xy[0].min()]] for x in contoursred]
    cv2.drawContours(img_s1, contoursred, -1, (255, 0, 0), 1)
    cv2.drawContours(img_s1, contoursblue, -1, (0, 0, 255), 1)
    # cv2.circle(img_s, (int(r)-n, int(r)-n), 3, (255, 0, 0), -1)
    # plt.imshow(img_s), plt.title('img_s'), plt.show()


    '''
    位置逻辑判断
    '''
    if len(contoursred) == 2:  # 如果识别出两处红标记
        area1 = cv2.contourArea(contoursred[0])
        area2 = cv2.contourArea(contoursred[1])
        if area1 > area2:
            linered = contoursred[0]
            pointred = contoursred[1]
        else:
            linered = contoursred[1]
            pointred = contoursred[0]
        # 根据红线区域点拟合直线
        line = cv2.fitLine(linered, cv2.DIST_L2, 0, 0.01, 0.01)
        angle = math.degrees(math.asin(line[1]))
        # 画出拟合红线的直线
        k = line[1] / line[0]
        b = line[3] - k * line[2]
        cv2.line(img_s1, (0, int(b)), (int(line[2]),int(line[3])),(0, 0, 255))
        # 计算小红点是否在红线延长线上
        rect1 = cv2.minAreaRect(linered)
        rect2= cv2.minAreaRect(pointred)
        points = [rect1[0],rect2[0]] # 得到两个标记得中心位置坐标
        points = np.array(points)
        linerelative = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01) # 拟合直线
        angles1 = math.degrees(math.asin(linerelative[1])) # 计算角度
        op = abs(abs(angle)-abs(angles1))
        if abs(op)>30:
            state = -1 # 松动
            print('夹角过大:', op)

        elif len(contoursblue) == 1: # 有且只有一个蓝色点
            rect3 = cv2.minAreaRect(contoursblue[0])
            # 判断距离
            d1 = cal_distance(rect1[0],rect2[0])
            d2 = cal_distance(rect1[0], rect3[0])
            if d1>d2:
                state = -1 # 松动
                print('红线距离蓝点比红点更近')
            else:
                state = 1 # 正常
                print('红线距离红点比蓝点更近')
        elif len(contoursblue) == 0:   # 没有识别出蓝色点
            # 利用检测圆心到红点距离来判断
            d1 = cal_distance(rect1[0], rect2[0])
            d3 = cal_distance((r-n, r-n), rect2[0])  # 圆心到红点距离
            #画出点的位置
            img2 = img_s.copy()
            print(rect1[0])
            cv2.circle(img2,(int(rect1[0][0]),int(rect1[0][1])), 1, (0, 0, 255), 3) # 红线 蓝色
            cv2.circle(img2, (int(rect2[0][0]),int(rect2[0][1])), 1, (255, 0, 0), 3) #红点 红色
            cv2.circle(img2,(int(r-n), int(r-n)), 1, ( 0, 255, 0), 3) #圆心 绿色
            plt.imshow(img2), plt.title('Test Image'), plt.show()
            if d3>d1-3:
                state = 1
                print(d1 - d3)
                print('红线到红点比圆心到红点距离更近')
            else:
                print(d1 - d3)
                state = -1
                print('圆心到红点比红线到红点距离更近')
        else:

            state = -2
            print('蓝点太多了我分不清')

    elif len(contoursred) == 1: # 只识别出redline
        if len(contoursblue) == 1: # 且识别出一个蓝色标记点
            #则判断角度，再判断距离
            linered = contoursred[0]
            pointblue = contoursblue[0]
            # 根据红线区域点拟合直线
            line = cv2.fitLine(linered, cv2.DIST_L2, 0, 0.01, 0.01)
            angle = math.degrees(math.asin(line[1]))
            # 计算小蓝点是否在红线延长线上
            rect1 = cv2.minAreaRect(linered)
            rect2 = cv2.minAreaRect(pointblue)
            points = [rect1[0], rect2[0]]  # 得到两个标记得中心位置坐标
            points = np.array(points)
            linerelative = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)  # 拟合直线
            angles1 = math.degrees(math.asin(linerelative[1]))  # 计算角度
            op = abs(abs(angle) - abs(angles1))
            if abs(op) > 30:
                state = -1  # 松动
                print('夹角过大:', op)
            else: #判断距离
                # 利用检测圆心到红点距离来判断
                d1 = cal_distance(rect1[0], rect2[0])  # 红线到蓝点距离
                d3 = cal_distance([r-n, r-n], rect2[0])  # 圆心到蓝点距离
                if d1 + 5 > d3:
                    state = 1
                    print('圆心比红线到蓝点距离更近')
                else:
                    state = -1
                    print('红线比圆心到蓝点距离更近')
        else: # 未识别出蓝色标记点
            state = -2 # 无法判断
            print('识别信息不足')
    elif len(contoursred) == 0: # 可增强提取算法
        state = -2
        print('红线在哪呢')
    else: # 可利用定位或面积判别
        state = -2
        print('好多红线呀我晕了')
    print('state:',state)
    cv2.imshow("out", img_s1)
    cv2.waitKey(0)

        # 画外接矩形
        # plot_image = draw_rbbox(img_s1, contoursred[1])
        # plt.imshow(plot_image), plt.title('plot_image'), plt.show()






