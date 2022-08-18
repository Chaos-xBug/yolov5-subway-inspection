import matplotlib.pyplot as plt
import numpy as np
import cv2
import os

def detect_line(img):
    # 进行均值滤波
    dst_img = cv2.blur(img, [3, 3])
    # plt.imshow(dst_img), plt.title('DST_Image'), plt.show()

    hsv_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)
    # plt.imshow(hsv_img), plt.title('HSV_Image'), plt.show()

    mask1 = cv2.inRange(hsv_img, np.array([0, 63, 33]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv_img, np.array([163, 80, 50]), np.array([180, 255, 255]))
    mask = mask1 | mask2
    # plt.imshow(mask, 'gray'), plt.title('Mask'), plt.show()

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    open =cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    close = cv2.morphologyEx(open, cv2.MORPH_CLOSE, kernel2)
    # plt.imshow(open), plt.title('Open'), plt.show()
    return close


basepath = r'E:\\04-Data\\02-robot\\04-Xingqiao0512\\KeyHoleCap\\Bolt\\'

for filename in os.listdir(basepath):
    # filename = basepath + 'KeyholeCap41_Pic_2022_05_13_150620_3.bmp'
    # filename = basepath + 'KeyholeCap40_Pic_2022_05_13_150722_7.bmp'
    print(filename)
    img = cv2.imread(os.path.join(basepath,filename))

    res = detect_line(img)

    # 显示图像
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 8), dpi=100)
    axes[0].imshow(img[:, :, ::-1])
    axes[0].set_title("原图")
    axes[1].imshow(res)
    axes[1].set_title("标记线检测后的图像")
    plt.show()
    # cv2.imshow("out", img)
    # cv2.waitKey(0)