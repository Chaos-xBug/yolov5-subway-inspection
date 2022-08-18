import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_rbbox(img, points):
    img = cv2.drawContours(img, [points], 0, (255, 0, 0), 2)
    rect = cv2.minAreaRect(points)
    print(rect)
    points_rect = cv2.boxPoints(rect)
    box = np.int0(points_rect)
    img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)
    return img


print(cv2.__version__)  # 4.5.1
img = np.zeros([500, 500, 3], dtype=np.uint8)
# 样例1
points1 = np.array([[100, 100], [180, 200], [140, 270], [60, 160]])
# 样例2
points2 = np.array([[320.4896, 306.2144],
                    [320.4896, 281.0297],
                    [381.2337, 281.0297],
                    [381.2337, 306.2144]], dtype=np.int64)
# 样例3
points3 = np.array([[222.4992, 345.3907],
                    [222.4992, 317.4076],
                    [281.2018, 317.4076],
                    [281.2018, 345.3907]], dtype=np.int64)
# 样例4
points4 = np.array([[450, 100], [480, 160], [200, 300], [150, 240]])

img = draw_rbbox(img, points1)
img = draw_rbbox(img, points2)
img = draw_rbbox(img, points3)
img = draw_rbbox(img, points4)
# cv2.imwrite(r'/home/111.jpg', img)
plt.imshow(img), plt.title('plot_image'), plt.show()
'''
    样例依次对应xywh-angle
    ((120.000, 185.0000), (157.735, 74.963), 51.3401)
    ((350.499, 293.499), (60.999, 24.999), 0.0)
    ((251.5, 331.0), (28.0, 59.0), 90.0)
    ((316.7883, 203.832), (75.515, 332.871), 64.9831)
'''