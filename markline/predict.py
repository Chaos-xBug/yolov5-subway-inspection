import copy
import cv2
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from markline.nets.BiSeNet import BiSeNetV2
import time
import matplotlib.pyplot as plt


def cnt_area(cnt):
    """
    ！！此方法为内部调用，无需关心！！
    这是用于OpenCV中轮廓从大到小排序的方法
    :param cnt: 轮廓
    :return: 轮廓
    """
    area = cv2.contourArea(cnt)
    return area


class SegmentationMetric(object):
    """
    ！！此方法为内部调用，无需关心！！
    这是计算mIoU的方法
    """

    def __init__(self, numClass):
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def meanIntersectionOverUnion(self):
        # Intersection = TP Union = TP + FP + FN
        # IoU = TP / (TP + FP + FN)
        intersection = np.diag(self.confusionMatrix)
        union = np.sum(self.confusionMatrix, axis=1) + np.sum(self.confusionMatrix, axis=0) - np.diag(
            self.confusionMatrix)
        IoU = intersection / union  # [背景,前景]两类
        mIoU = np.nanmean(IoU)
        return IoU[1]

    def genConfusionMatrix(self, imgPredict, imgLabel):
        # remove classes from unlabeled pixels in gt image and predict
        mask = (imgLabel >= 0) & (imgLabel < self.numClass)
        label = self.numClass * imgLabel[mask] + imgPredict[mask]
        count = np.bincount(label, minlength=self.numClass ** 2)
        confusionMatrix = count.reshape(self.numClass, self.numClass)
        return confusionMatrix

    def addBatch(self, imgPredict, imgLabel):
        assert imgPredict.shape == imgLabel.shape
        self.confusionMatrix += self.genConfusionMatrix(imgPredict, imgLabel)


class maskobject(object):
    """
    此类用于红蓝标记线检测，含有标记线提取（图像分割）、寻找标记线轮廓、标记线松动状态的三种方法。
    """
    # 默认参数设置
    _defaults = {
        "model_path": 'C:/Users/Lenovo/PycharmProjects/yolov5-wjd/markline/model/bestbise0824.pth',  # 模型权重存放地址
        "model_image_size": (224, 224, 3),  # 网络输入尺寸
        "num_classes": 3  # 类别+1
    }

    def __init__(self, **kwargs):
        """
        这是初始化函数，初始化参数并加载模型，默认参数见 _defaults
        :param kwargs:
        """
        self.__dict__.update(self._defaults)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = BiSeNetV2(n_classes=self.num_classes).eval()
        state_dict = torch.load(self.model_path, map_location=self.device)
        self.net.load_state_dict(state_dict)
        # print(self.device==torch.device("cuda"))
        if self.device == torch.device("cuda"):
            self.net = nn.DataParallel(self.net)
        self.net = self.net.to(self.device)
        # mask可视化赋值颜色对应类别+1
        self.colors = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (0, 0, 128)]

    def letterbox_image(self, image, size):
        """
        ！！此方法为内部调用，无需关心！！
        防止输入图片在resize过程变形，在图片周围加黑条（以长边为准，补齐短边）
        :param image: 原始图片
        :param size: 模型输入需要的size
        :return: new_img:调整后的图片,nw：新的宽,nh：新的高
        """
        image = image.convert("RGB")
        iw, ih = image.size
        w, h = size
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', size, (0, 0, 0))
        new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))
        return new_image, nw, nh

    def detect_image(self, image):
        """
        这是基于图像分割的标记线检测方法，能够检测出输入图像中的标记线并以轮廓显示、蒙版等形式返回。
        输入与输出的图片均为OpenCV格式。
        :param image: 未做任何处理的OpenCV格式图片
        :return: old_img：将标记线轮廓显示在原始图片上, image_mask：标记线的mask形式, contours：标记线的全部轮廓,
        img_cut：含所有标记线的最小区域的图片（所有轮廓的最大外接矩），gray_cut：含所有标记线的最小区域的灰度图，后续开发使用
        """
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(np.uint8(image))
        old_img = np.asarray(copy.deepcopy(image))  # 对输入图像进行一个备份，后面用于绘图
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # 进行不失真的resize，添加黑条，进行图像归一化
        image, nw, nh = self.letterbox_image(image, (self.model_image_size[1], self.model_image_size[0]))
        images = [np.array(image) / 255]
        images = np.transpose(images, (0, 3, 1, 2))
        with torch.no_grad():
            self.net.eval()
            images = torch.from_numpy(images).type(torch.FloatTensor)
            images = images.to(self.device)
            pr = self.net(images)[0][0]
            #   取出每一个像素点的种类
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy().argmax(axis=-1)
            #   将黑条部分截取掉
            pr = pr[int((self.model_image_size[0] - nh) // 2):int((self.model_image_size[0] - nh) // 2 + nh),
                 int((self.model_image_size[1] - nw) // 2):int((self.model_image_size[1] - nw) // 2 + nw)]
        #   创建一副新图，并根据每个像素点的种类赋予颜色
        seg_img = np.zeros((np.shape(pr)[0], np.shape(pr)[1], 3))
        for c in range(self.num_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (self.colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (self.colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (self.colors[c][2])).astype('uint8')
        #   将新图片转换成Image的形式
        image_mask = Image.fromarray(np.uint8(seg_img)).resize((orininal_w, orininal_h), Image.NEAREST)
        # 在原图片上画上轮廓
        gray = cv2.cvtColor(np.asarray(image_mask), cv2.COLOR_BGR2GRAY)  # R:15,B:75
        # print(gray.shape())
        gray_cutcontours, hierarchy = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv_contours = []
        for contour in gray_cutcontours:
            area = cv2.contourArea(contour)
            if area <= 120:
                cv_contours.append(contour)
            else:
                continue
        cv2.fillPoly(gray, cv_contours, 0)
        xy = np.where(gray > 0)
        if gray.max() < 1 or (xy[0].max() - xy[0].min()) < 3 or (xy[1].max() - xy[1].min()) < 3:
            gray_cut = gray
            img_cut = old_img
        else:
            gray_cut = gray[xy[0].min(): xy[0].max(), xy[1].min():xy[1].max()]  # 含所有标记线的最小区域
            img_cut = old_img[xy[0].min(): xy[0].max(), xy[1].min():xy[1].max()]
        red = copy.deepcopy(gray_cut)  # 红色标记线R:15
        blue = copy.deepcopy(gray_cut)  # 蓝色标记线B:75
        red[red < 10] = red[red > 20] = 0
        blue[blue < 70] = blue[blue > 80] = 0
        # 轮廓处理，寻找红蓝标记线并展示在old_img上
        contoursred, hierarchy = cv2.findContours(red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursblue, hierarchy_ = cv2.findContours(blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contoursredshow = [x + [[xy[1].min(), xy[0].min()]] for x in contoursred]
        cv2.drawContours(old_img, contoursredshow, -1, (255, 0, 0), 1)
        contoursblueshow1 = [x + [[xy[1].min(), xy[0].min()]] for x in contoursblue]
        cv2.drawContours(old_img, contoursblueshow1, -1, (0, 0, 255), 1)
        image_mask = np.asarray(image_mask)
        # contours = contoursred + contoursblue  # 后期处理过程中，无需区分红蓝标记线
        contours = contoursred
        old_img = cv2.cvtColor(old_img, cv2.COLOR_RGB2BGR)
        img_cut = cv2.cvtColor(img_cut, cv2.COLOR_RGB2BGR)

        return old_img, image_mask, contours, img_cut, gray_cut

    def findline(self, image):
        """
        ！！此方法为内部调用，无需关心！！
        这是标记线轮廓解析方法，提取有效轮廓的角度、中心点等信息，用于后续的判断是否松动
        :param image:未做任何处理的OpenCV格式图片
        :return:angles：所有有效轮廓线的角度集合, point：所有有效轮廓线的中心点集合, contours：所有轮廓线,
        old_img：将标记线轮廓显示在原始图片上, image_mask：标记线的mask形式,
        """
        old_img, image_mask, contours, img_cut, gray_cut = self.detect_image(image)
        # contours.sort(key=cnt_area, reverse=True)
        angles = []
        point = []
        for c in contours:
            rect = cv2.minAreaRect(c)
            area = cv2.contourArea(c)
            if area < 40:
                continue
            line = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            # line的形式是[[cos  a],[sin a],[point_x],[point_y]], 前面两项是有关直线与Y正半轴（这里指的是屏幕坐标系）夹角a的三角函数
            angle = math.degrees(math.asin(line[1]))
            # 图像预处理，筛选轮廓，剔除长宽比小于1.5的轮廓
            if rect[1][0] / rect[1][1] > 1.5 or rect[1][1] / rect[1][0] > 1.5:
                angles.append(angle)
            point.append([line[2][0], line[3][0]])
        return angles, point, contours, old_img, image_mask, gray_cut

    def linestate(self, image1, image2):
        """
        这是根据标记线的特征判断是否松动的方法，输入历史与当前的图片，
        输出当前零部件的状态（-3:未检测到轮廓，-2:未检测到有效轮廓，-1:轮廓个数与历史不符，0:未松动，1:松动）
        :param image1:历史图片（OpenCV）
        :param image2:当前图片（OpenCV）
        :return:state：标记线的状态:（-3,-2,-1,0,1）, old_img：将标记线轮廓显示在原始图片上, image_mask：标记线的mask形式, old_img1, image_mask1
        """
        angles, point, contours, old_img, image_mask, gray_cut = self.findline(image1)
        angles1, point1, contours1, old_img1, image_mask1, gray_cut1 = self.findline(image2)

        # ----------------以下为角度和坐标判别---------------------#
        # print("angles0:", angles, "angles1:", angles1)
        if len(contours) == 0 or len(contours1) == 0:
            state = -3
        else:
            if len(angles) == len(angles1) > 0:
                op = np.linalg.norm(np.array(angles) - np.array(angles1))
                print("angledis:", op)
                if op < 20 or op > 160:  # 欧氏距离角度偏差小于20或者(160=180-20)
                    stateangle = 0
                else:
                    stateangle = 1
            elif len(angles) != len(angles1):
                stateangle = -1
            else:
                stateangle = -2
            # print("point0:", point, "point1:", point1)
            if len(point) == len(point1) > 0:
                dis = np.linalg.norm(np.array(point) - np.array(point1))
                print("pointdis:", dis)
                if dis < 20:  # 欧氏距离位置坐标偏差小于20pix
                    statepoint = 0
                else:
                    statepoint = 1
            elif len(point) != len(point1):
                statepoint = -1
            else:
                statepoint = -2
            # ----------------!判断是否松动!---------------------#
            if stateangle == 0 and statepoint == 0:
                state = 0  # 正常
            elif stateangle == 1 or statepoint == 1:
                state = 1  # 松动
            elif statepoint == -1:
                if len(point) < len(point1):
                    state = 1  # 松动（当前轮廓比历史记录多时，认为松动）
                else:
                    state = -1  # 轮廓与历史不符（特指当前轮廓小于历史轮廓）
            else:
                state = -2  # 轮廓过小，无法用于判断

        return state, old_img, image_mask, old_img1, image_mask1

    def line_iou(self, image, image1):
        """
         这是根据标记线的IOU（重叠程度）判断是否松动的方法，输入历史与当前的图片，
        输出当前零部件的状态（-3:未检测到轮廓，0:未松动，1:松动）
        :param image:历史图片（OpenCV）
        :param image1:当前图片（OpenCV）
        :return:state：标记线的状态:（-3,0,1）, old_img：将标记线轮廓显示在原始图片上, image_mask：标记线的mask形式, old_img1, image_mask1
        """
        old_img, image_mask, contours, img_cut, gray_cut = self.detect_image(image)
        old_img1, image_mask1, contours1, img_cut1, gray_cut1 = self.detect_image(image1)
        gray_cut = cv2.resize(gray_cut, (100, 100), interpolation=cv2.INTER_NEAREST)
        gray_cut1 = cv2.resize(gray_cut1, (100, 100), interpolation=cv2.INTER_NEAREST)
        gray_cut[gray_cut > 0] = 1
        gray_cut1[gray_cut1 > 0] = 1
        metric = SegmentationMetric(2)  # 2表示有2个分类，背景与前景
        metric.addBatch(gray_cut, gray_cut1)
        mIoU = metric.meanIntersectionOverUnion()
        print("mIoU:", mIoU)
        if gray_cut.max() == 0 or gray_cut1.max() == 0:
            mIoU_state = -3
        elif mIoU > 0.6:
            mIoU_state = 0
        else:
            mIoU_state = 1
        return mIoU_state, old_img, image_mask, old_img1, image_mask1

    def doublestate(self, image1, image2):
        """
        这是根据标记线的IOU（先）和特征（后）判断是否松动的方法，输入历史与当前的图片，
        输出当前零部件的状态（-3:未检测到轮廓，-2:未检测到有效轮廓，-1:轮廓个数与历史不符，0:未松动，1:松动）
        :param image1:历史图片（OpenCV）
        :param image2:当前图片（OpenCV）
        :return:state：标记线的状态:（-3,-2,-1,0,1）, old_img：将标记线轮廓显示在原始图片上, image_mask：标记线的mask形式, old_img1, image_mask1
        """
        s0 = time.time()
        angles, point, contours, old_img, image_mask, gray_cut = self.findline(image1)
        angles1, point1, contours1, old_img1, image_mask1, gray_cut1 = self.findline(image2)
        e0 = time.time()
        # print("串联:", e0 - s0)
        # s = time.time()
        # ls=[image1, image2]
        # res = Pool(2).map(self.findline, ls)
        #
        # e = time.time()
        # print("多进程:", e - s)
        # ----------------以下为IOU判别---------------------#
        gray_cut = cv2.resize(gray_cut, (100, 100), interpolation=cv2.INTER_NEAREST)
        gray_cut1 = cv2.resize(gray_cut1, (100, 100), interpolation=cv2.INTER_NEAREST)
        gray_cut[gray_cut > 0] = 1
        gray_cut1[gray_cut1 > 0] = 1
        metric = SegmentationMetric(2)  # 2表示有2个分类，背景与前景
        metric.addBatch(gray_cut, gray_cut1)
        mIoU = metric.meanIntersectionOverUnion()
        if gray_cut.max() == 0 or gray_cut1.max() == 0:
            state = -3
        elif mIoU > 0.6:
            state = 0
        else:
            state = 1
            # ----------------以下为角度和坐标判别---------------------#
            # print("angles0:", angles, "angles1:", angles1)
            if len(contours) == 0 or len(contours1) == 0:
                state = -3  # 未检测到轮廓
            else:
                if len(angles) == len(angles1) > 0:
                    op = np.linalg.norm(np.array(angles) - np.array(angles1))
                    print("angledis:", op)
                    if op < 20 or op > 160:  # 欧氏距离角度偏差小于20
                        stateangle = 0
                    else:
                        stateangle = 1
                elif len(angles) != len(angles1):
                    stateangle = -1
                else:
                    stateangle = -2
                # print("point0:", point, "point1:", point1)
                if len(point) == len(point1) > 0:
                    # ----------欧氏距离----------------#
                    dis = np.linalg.norm(np.array(point) - np.array(point1))
                    print("pointdis:", dis)
                    if dis < 20:  # 欧氏距离位置坐标偏差小于20pix
                        statepoint = 0
                    else:
                        statepoint = 1
                elif len(point) != len(point1):
                    statepoint = -1
                else:
                    statepoint = -2
                # ----------------!判断是否松动!---------------------#
                if stateangle == 0 and statepoint == 0:
                    state = 0  # 正常
                elif stateangle == 1 or statepoint == 1:
                    state = 1  # 松动
                elif statepoint == -1:
                    if len(point) < len(point1):
                        state = 1  # 松动（当前轮廓比历史记录多时，认为松动）
                    else:
                        state = -1  # 轮廓与历史不符（特指当前轮廓小于历史轮廓）
                else:
                    state = -2  # 轮廓过小，无法用于判断

        return state, old_img, image_mask, old_img1, image_mask1


def _imread(file_path):
    cv_img = cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), 1)
    return cv_img


def main():
    mask = maskobject()
    i = 0
    # while i < 10:
    image = _imread("C:/Users/Lenovo/PycharmProjects/yolov5-wjd/data/Line3/manques/11.bmp/1-Bolt-check.jpg")
    image1 = _imread("C:/Users/Lenovo/PycharmProjects/yolov5-wjd/data/Line3/manques/11.bmp/1-Bolt-check.jpg")
    plt.imshow(image), plt.title('111'), plt.show()
    timestrt = time.time()
    # 以下三种判别方式返回值一致，state, old_img, image_mask, old_img1, image_mask1
    # ----------角度、坐标判别法------------#
    # state = mask.linestate(image, image1)
    # ------------IoU判别法----------------#
    # state = mask.line_iou(image, image1)
    # -----先IoU判别法，后角度、坐标判别法----#

    state = mask.doublestate(image, image1)
    # print('state:',state)
    out = cv2.imread("C:/Users/Lenovo/PycharmProjects/yolov5-wjd/data/Line3/manques/11.bmp/1-Bolt-check.jpg")
    plt.imshow(out), plt.title('111'), plt.show()
    r_image = mask.detect_image(out)[0]
    cv2.namedWindow("out", cv2.WINDOW_KEEPRATIO)
    cv2.imshow("out", r_image)
    cv2.waitKey(0)
    print("状态：{},耗时：{:.3f}秒".format(state[0], time.time() - timestrt))

    i += 1


if __name__ == '__main__':

    main()
