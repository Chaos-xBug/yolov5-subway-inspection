import os
import random
from random import shuffle
import cv2
import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader


def is_even(a):
    if a % 2 == 0:
        a = a
    elif a % 2 == 1:
        a += 1
    return a

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a

class SiameseDataset(Dataset):
    def __init__(self, input_shape, dataset_path, train_ratio, train=True):
      super(SiameseDataset, self).__init__()

      self.dataset_path = dataset_path
      self.image_height = input_shape[0]
      self.image_width = input_shape[1]
      self.channel = input_shape[2]
      self.train_ratio = train_ratio
      self.train = train

      self.num_train = 0
      self.num_val = 0
      self.num_same_train = 0
      self.num_diff_train = 0
      self.num_same_val = 0
      self.num_diff_val = 0

      self.train_lines = []
      self.train_labels = []
      self.val_lines = []
      self.val_labels = []

      self.same_train_lines = []
      self.same_train_labels = []
      self.diff_train_lines = []
      self.diff_train_labels = []

      self.load_dataset()

    def __len__(self):
      if self.train:
        return self.num_train
      else:
        return self.num_val

    def load_dataset(self):
      random.seed(1)
      print('运行了load_dataset')
      for type_folder in os.listdir(self.dataset_path):
          if type_folder == "same":
              image_folder = os.path.join(self.dataset_path, type_folder)
              i = 0
              for image in os.listdir(image_folder):
                  self.same_train_lines.append(os.path.join(image_folder, image))
                  self.same_train_labels.append(i)
                  i += 1
          elif type_folder == "diff":
              image_folder = os.path.join(self.dataset_path, type_folder)
              j = 0
              for image in os.listdir(image_folder):
                  self.diff_train_lines.append(os.path.join(image_folder, image))
                  self.diff_train_labels.append(j)
                  j += 1
      print("排序前：", self.same_train_lines[0:5])
      self.same_train_lines.sort(key=lambda x: int(x[14:-10]))
      print("排序后：", self.same_train_lines[0:20])

      print("排序前：", self.diff_train_lines[0:5])
      self.diff_train_lines.sort(key=lambda x: int(x[14:-10]))
      print("排序后：", self.diff_train_lines[0:20])
      
      print("same文件夹下的文件数量：", len(self.same_train_lines))
    
      print("diff文件夹下的文件数量：", len(self.diff_train_lines))
      
      # 这里生成了两个list
      # 第一个list里是same文件夹下的所有图片
      # 第二个list里是diff文件夹下的所有图片
      # 两个list里的图片数量加总就是所有数据的数量

      #  1. 计算 训练集的个数 验证集的个数
      total_sample = int(len(self.same_train_lines) + len(self.diff_train_lines))
      # 训练集个数 要求一定是偶数
      self.num_train = int(is_even(total_sample * self.train_ratio))
      # 验证集个数： 总数量-训练集个数
      self.num_val = total_sample - self.num_train
      print("文件总数：", total_sample)
      print("训练集数量：", self.num_train)
      print("验证集数量：", self.num_val)
      print("---------------------------------------------------------------------")

      #  2. 训练集由：若干对 相同的一对（label为1） 和 不同的一对（label为0）组成
      # 由于训练集是按总样本的ratio来生成训练集
      # 那么可以理解为： same文件夹下的文件*ratio（偶数） + diff文件夹下的文件*ratio（偶数） 组成了总的训练集
      # same文件夹下的文件数量*ratio （N）

      # same文件夹中共有这么多对数据
      same_pair = len(self.same_train_lines) // 2

      self.num_same_train = int(is_even(int(len(self.same_train_lines) * self.train_ratio)))
      # same_pair = self.num_same_train // 2

      # diff文件夹下的文件数量*ratio （M）
      self.num_diff_train = self.num_train - self.num_same_train
      # diff文件下下共有这么多对数据
      diff_pair = len(self.diff_train_lines) // 2

      self.num_same_val = len(self.same_train_lines) - self.num_same_train
      self.num_diff_val = self.num_val - self.num_same_val

      # same文件夹的处理
      # 随机排序same_pair个数据，生成一个索引的列表 * 2 即为全部为偶数的索引的列表
      # 上面的列表元素全部+1即为对应的奇数的索引的列表

      a = range(0, same_pair)
      b = random.sample(a, same_pair)

      idx_even = []
      idx_single = []
      for i in b:
          i *= 2
          idx_even.append(i)
          i += 1
          idx_single.append(i)
      print("偶数的索引", idx_even)
      print("奇数的索引", idx_single)

      idx = []
      for i in range(len(idx_even)):
          idx.append(idx_even[i])
          idx.append(idx_single[i])
      print("新same文件夹的索引：", idx)

      new_train_list = []
      for i in idx:
          new_train_list.append(self.same_train_lines[i])
      # print(new_train_list)
      for i in range(self.num_same_train):
          self.train_lines.append(new_train_list[i])

      for i in self.same_train_lines:
          if self.train_lines.count(i) < 1:
              self.val_lines.append(i)
      for i in range(8):
        print(self.train_lines[i])
      # 68, 69, 290, 291, 432, 433, 410, 411,
      print(self.same_train_lines[68])
      print(self.same_train_lines[69])
      print(self.same_train_lines[290])
      print(self.same_train_lines[291])
      print(self.same_train_lines[432])
      print(self.same_train_lines[433])
      print(self.same_train_lines[410])
      print(self.same_train_lines[411])
    

      # diff文件的处理
      even_idx = []
      single_idx = []

      c = range(0, diff_pair)
      d = random.sample(c, diff_pair)

      for i in d:
          i *= 2
          even_idx.append(i)
          i += 1
          single_idx.append(i)
      print("偶数的索引", even_idx)
      print("奇数的索引", single_idx)

      idx1 = []
      for i in range(len(even_idx)):
          idx1.append(even_idx[i])
          idx1.append(single_idx[i])
      print("新diff文件夹的索引：", idx1)
      new_train_list_diff = []
      for i in idx1:
          new_train_list_diff.append(self.diff_train_lines[i])
      # print(new_train_list)
      for i in range(self.num_diff_train):
          self.train_lines.append(new_train_list_diff[i])

      for i in self.diff_train_lines:
          if self.train_lines.count(i) < 1:
              self.val_lines.append(i)

      for i in range(8):
        print(self.val_lines[i])

      print("训练集有", len(self.train_lines), "个，其中前", self.num_same_train, "个样本是same的，diff的样本有", self.num_diff_train)
      print("验证集有", len(self.val_lines), "个，其中前", self.num_same_val, "个样本是same的，diff的样本有", self.num_diff_val)
      print("----------------------------------------------------")

      # 关于label， 不重要这里

      new_label_list = []
      for i in idx:
          new_label_list.append(self.same_train_labels[i])
      
      for i in range(self.num_same_train):
          self.train_labels.append(new_label_list[i])

      for i in self.same_train_labels:
          if self.train_labels.count(i) < 1:
              self.val_labels.append(i)

      # diff文件的处理

      new_label_list_diff = []
      for i in idx1:
          new_label_list_diff.append(self.diff_train_labels[i])
      # print(new_train_list)
      for i in range(self.num_diff_train):
          self.train_labels.append(new_label_list_diff[i])

      for i in self.diff_train_labels:
          if self.train_labels.count(i) < 1:
              self.val_labels.append(i)

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, flip_signal=False):
        if self.channel == 1:
            image = image.convert("RGB")

        h, w = input_shape
        #------------------------------------------#
        #   图像大小调整
        #------------------------------------------#
        rand_jit1   = rand(1-jitter,1+jitter)
        rand_jit2   = rand(1-jitter,1+jitter)
        new_ar      = w/h * rand_jit1/rand_jit2

        scale = rand(0.75,1.25)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        
        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = rand()<.5
        if flip and flip_signal: 
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        
        #------------------------------------------#
        #   放置图像
        #------------------------------------------#
        dx = int(rand(0, w-nw))
        dy = int(rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (255,255,255))

        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   图像旋转
        #------------------------------------------#
        rotate = rand()<.5
        if rotate: 
            angle = np.random.randint(-5,5)
            a,b = w/2,h/2
            M = cv2.getRotationMatrix2D((a,b),angle,1)
            image = cv2.warpAffine(np.array(image), M, (w,h), borderValue = [255,255,255]) 

        #------------------------------------------#
        #   色域扭曲
        #------------------------------------------#
        hue = rand(-hue, hue)
        sat = rand(1, sat) if rand()<.5 else 1/rand(1, sat)
        val = rand(1, val) if rand()<.5 else 1/rand(1, val)
        x = cv2.cvtColor(np.array(image,np.float32)/255, cv2.COLOR_RGB2HSV)
        x[..., 0] += hue*360
        x[..., 0][x[..., 0]>1] -= 1
        x[..., 0][x[..., 0]<0] += 1
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:,:, 0]>360, 0] = 360
        x[:, :, 1:][x[:, :, 1:]>1] = 1
        x[x<0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB)*255

        if self.channel == 1:
            image_data = Image.fromarray(np.uint8(image_data)).convert("L")
        return image_data


    def _convert_path_list_to_images_and_labels(self, path_list):
        # print("运行了convert")
        # 这里传入训练集中的图片的路径
        # 共有number_of_pairs对数据
        number_of_pairs = int(len(path_list) / 2)
        # 前 same_pairs 对数据返回的label是1
        num_same = self.num_same_train
        same_pairs = int(num_same / 2)
        # 定义网络的输入和标签
        pairs_of_images = [np.zeros((number_of_pairs, self.channel, self.image_height, self.image_width)) for i in range(2)]
        labels = np.zeros((number_of_pairs, 1))
        # 现在的pairs_of_images是一个列表，对应传入的一对图片
        # 列表的每个元素是一个（number_of_pairs， 3， 105， 105）维的array
        # labels就是一个（number_of_pairs, 1）维的数组，值为0
        for pair in range(number_of_pairs):
            # 将图片填充到输入1中
            image1 = Image.open(path_list[pair * 2])
            image1 = self.get_random_data(image1, [self.image_height, self.image_width])
            # image1 = image1.resize((self.image_width, self.image_height))
            image1 = np.asarray(image1).astype(np.float64)
            image1 = np.transpose(image1, [2, 0, 1])
            image1 = image1 / 255
            if self.channel == 1:
                pairs_of_images[0][pair, 0, :, :] = image1
            else:
                pairs_of_images[0][pair, :, :, :] = image1
            # 将图片填充到输入2中
            image2 = Image.open(path_list[pair * 2 + 1])
            image2 = self.get_random_data(image2, [self.image_height, self.image_width])
            # image2 = image2.resize((self.image_width, self.image_height))
            image2 = np.asarray(image2).astype(np.float64)
            image2 = np.transpose(image2, [2, 0, 1])
            image2 = image2 / 255
            if self.channel == 1:
                pairs_of_images[1][pair, 0, :, :] = image2
            else:
                pairs_of_images[1][pair, :, :, :] = image2

            # 元素0,1对应label=1， 元素2,3对应label=0
            if (pair + 1) % 2 == 0:
                labels[pair] = 0
            else:
                labels[pair] = 1

        return pairs_of_images, labels

    def __getitem__(self, index):

        # print("运行了getitem")
          # 这个方法中主要要做 生成一个path_list传入上面的两个方法中的任意一个， 生成images和labels
        batch_image_path = []
        if self.train:
            lines = self.train_lines
            labels = self.train_labels
            # 逻辑是：
            # 把lines分成same和diff两个部分，每个部分是偶数数量个元素，在两个部分分别进行下一步操作
            # 在0~len(same_lines)的范围内随机选择一个偶数，在same_lines中选取下标为这个偶数的元素和下标为这个偶数+1的元素仿佛bact_image_path中
            # 在0~len(diff_lines)的范围内随机选择一个偶数，在diff_lines中选取下标为这个偶数的元素和下标为这个偶数+1的元素仿佛bact_image_path中
            # 将batch_image_path传入_convert_to_images_labels方法中， return pairs_of_images， labels
            same_lines = self.same_train_lines
            diff_lines = self.diff_train_lines

            idx_1 = random.randrange(0, self.num_same_train, 2)
            batch_image_path.append(same_lines[idx_1])
            batch_image_path.append(same_lines[idx_1 + 1])

            idx_2 = random.randrange(0, self.num_diff_train, 2)
            batch_image_path.append(diff_lines[idx_2])
            batch_image_path.append(diff_lines[idx_2 + 1])

            images, labels = self._convert_path_list_to_images_and_labels(batch_image_path)

        else:
            lines = self.val_lines
            labels = self.val_labels

            same_lines = lines[:self.num_same_val]
            diff_lines = lines[self.num_same_val:]

            idx_1 = random.randrange(0, self.num_same_val, 2)
            batch_image_path.append(same_lines[idx_1])
            batch_image_path.append(same_lines[idx_1 + 1])

            idx_2 = random.randrange(0, self.num_diff_val, 2)
            batch_image_path.append(diff_lines[idx_2])
            batch_image_path.append(diff_lines[idx_2 + 1])

            images, labels = self._convert_path_list_to_images_and_labels(batch_image_path)

        return images, labels


# DataLoader中collate_fn使用
def dataset_collate(batch):
    left_images = []
    right_images = []
    labels = []
    for pair_imgs, pair_labels in batch:
        for i in range(len(pair_imgs[0])):
            left_images.append(pair_imgs[0][i])
            right_images.append(pair_imgs[1][i])
            labels.append(pair_labels[i])

    return np.array([left_images, right_images]), np.array(labels)





