import cv2
import os
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

def preprocess_image(CV2im, resize_im=True):
    """
        Processes image for CNNs
    Args:
        PIL_img (PIL_img): Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        im_as_var (Pytorch variable): Variable that contains processed float tensor
    """
    # mean and std list for channels (Imagenet)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        CV2im = cv2.resize(CV2im, (224, 224))
    im_as_arr = np.float32(CV2im)
    im_as_arr = np.ascontiguousarray(im_as_arr[..., ::-1])
    im_as_arr = im_as_arr.transpose(2, 0, 1) # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(im_as_arr):
        im_as_arr[channel] /= 255
        im_as_arr[channel] -= mean[channel]
        im_as_arr[channel] /= std[channel]
    # Convert to float tensor
    im_as_ten = torch.from_numpy(im_as_arr).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    im_as_ten.unsqueeze_(0)
    # Convert to Pytorch variable
    im_as_var = Variable(im_as_ten, requires_grad=True)
    return im_as_var

class FeatureVisualization():
    def __init__(self,selected_layer):
        self.selected_layer=selected_layer
        self.pretrained_model = models.vgg19(pretrained=True).features

    def process_image(self, img_path):
        img=cv2.imread(img_path)
        img=preprocess_image(img)
        return img

    def get_feature(self, img_path):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image(img_path)
        # print(input.shape)
        x=input
        res = []
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            if (index in self.selected_layer):
                res.append(x)
            # 可以条件提前终止
        return res

    def get_single_feature(self,img_path):
        features=self.get_feature(img_path)
        print(len(features))
        print(features[0].shape)

        feature0 = features[0][:,0,:,:]
        feature1 = features[1][:,0,:,:]
        feature2 = features[2][:,0,:,:]
        feature3 = features[3][:,0,:,:]

        # block3_pool_features = get_activations(base_model, 12, x)
        # block4_pool_features = get_activations(base_model, 16, x)

        # feature=features[:,0,:,:]
        # print(feature0.shape)
        feature00=feature0.view([feature0.shape[1],feature0.shape[2]]).data.numpy()
        feature11=feature1.view(feature1.shape[1],feature1.shape[2]).data.numpy()
        feature22=feature2.view(feature2.shape[1],feature2.shape[2]).data.numpy()
        feature33=feature3.view(feature3.shape[1],feature3.shape[2]).data.numpy()
        print(feature00.shape)

        feature00 = cv2.resize(feature00, [224, 224])
        feature11 = cv2.resize(feature11, [224, 224])
        feature22 = cv2.resize(feature22, [224, 224])
        feature33 = cv2.resize(feature33, [224, 224])
        print(feature00.shape)

        x0 = np.expand_dims(feature00,axis=2)
        x1 = np.expand_dims(feature11,axis=2)
        x2 = np.expand_dims(feature22,axis=2)
        x3 = np.expand_dims(feature33,axis=2)
        print(x0.shape)

        F = np.concatenate([x0, 2 * x1, x2, x3], 2)
        print(F.shape)
        # #use sigmod to [0,1]
        # F= 1.0/(1+np.exp(-1*F))
        # # to [0,255]
        # F=np.round(F*255)
        return F

    def save_feature_to_img(self, img_path1, img_path2):
        #to numpy
        F1=self.get_single_feature(img_path1) # Features from image patch 1
        F1 = np.square(F1)
        F2 = self.get_single_feature(img_path2)  # Features from image patch 1
        F2 = np.square(F2)
        d = np.subtract(F1, F2)
        d = np.square(d)
        d = np.sum(d, axis=3)
        plt.imshow(d), plt.show()




if __name__=='__main__':
    # get class
    myClass=FeatureVisualization([5, 9, 11, 15])
    # print (myClass.pretrained_model)
    path_base = r'E:\\09-Others\\Try\\'
    img_path1 =  os.path.join(path_base, r"T\\7.bmp")
    img_path2 = os.path.join(path_base, r"N\\7.jpg")
    myClass.save_feature_to_img(img_path1, img_path2)