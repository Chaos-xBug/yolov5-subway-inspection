import math
from PIL import Image
import numpy as np


def cal_pt_distance(pt1, pt2):

    dist = math.sqrt(pow(pt1[0]-pt2[0],2) + pow(pt1[1]-pt2[1],2))
    return dist

def letterbox_image(image, size):
    """
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


def gray_cut(close):
    xy = np.where(close > 0)
    gray = close[xy[0].min(): xy[0].max(), xy[1].min():xy[1].max()]

    return gray
