import cv2
import numpy as np
import torch
from torch.autograd import Variable
from torchvision import models
import matplotlib.pyplot as plt

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
    def __init__(self,img_path,selected_layer):
        self.img_path=img_path
        self.selected_layer=selected_layer
        self.pretrained_model = models.vgg19(pretrained=True).features

    def process_image(self):
        img=cv2.imread(self.img_path)
        img=preprocess_image(img)
        return img

    def get_feature(self):
        # input = Variable(torch.randn(1, 3, 224, 224))
        input=self.process_image()
        # print(input.shape)
        x=input
        for index,layer in enumerate(self.pretrained_model):
            x=layer(x)
            if (index == self.selected_layer):
                return x

    def get_single_feature(self):
        features=self.get_feature()
        print(features.shape)
        feature=features[:,0,:,:]
        print(feature.shape)
        feature=feature.view(feature.shape[1],feature.shape[2])
        print(feature.shape)
        return feature

    def save_feature_to_img(self):
        #to numpy
        feature=self.get_single_feature()
        feature=feature.data.numpy()
        #use sigmod to [0,1]
        feature= 1.0/(1+np.exp(-1*feature))
        # to [0,255]
        feature=np.round(feature*255)
        print(feature[0])
        plt.imshow(feature), plt.show()
        # cv2.imwrite('./img.jpg',feature)
        feature = cv2.resize(feature, [224, 224])
        return feature


#
# # Function to retrieve features from intermediate layers
# def get_activations(model, layer_idx, X_batch):
#     get_activations = K.function([model.layers[0].input], [model.layers[layer_idx].output])
#     activations = get_activations([X_batch, 0])
#     return activations
#
#
# # Function to extract features from intermediate layers
# def extra_feat(img, base_model):
#     x = image.img_to_array(img)
#     x = np.expand_dims(x, axis=0)
#     x = preprocess_input(x)
#     block1_pool_features = get_activations(base_model, 6, x)
#     block2_pool_features = get_activations(base_model, 11, x)
#     block3_pool_features = get_activations(base_model, 12, x)
#     block4_pool_features = get_activations(base_model, 16, x)
#     # block5_pool_features = get_activations(base_model, 12, x)
#
#     x1 = tf.image.resize(block1_pool_features[0], [224, 224])
#     x2 = tf.image.resize(block2_pool_features[0], [224, 224])
#     x3 = tf.image.resize(block3_pool_features[0], [224, 224])
#     x4 = tf.image.resize(block4_pool_features[0], [224, 224])
#     # x5 = tf.image.resize(block5_pool_features[0], [112, 112])
#
#     F = tf.concat([x1, x2, 2*x3, x4], 3)
#     return F
def get_feature(p):
    myClass0=FeatureVisualization(p,3)
    x0 = myClass0.save_feature_to_img()
    myClass1 = FeatureVisualization(p, 5)
    x1 = myClass1.save_feature_to_img()
    myClass2=FeatureVisualization(p,7)
    x2 = myClass2.save_feature_to_img()
    myClass3 = FeatureVisualization(p, 9)
    x3 = myClass3.save_feature_to_img()
    # x0 = np.expand_dims(feature00, axis=2)
    # x1 = np.expand_dims(feature11, axis=2)
    # x2 = np.expand_dims(feature22, axis=2)
    # x3 = np.expand_dims(feature33, axis=2)
    F = x0 + 2*x1 + x2 + x3
    # np.concatenate([x0, 2 * x1, x2, x3], 2)
    return F
if __name__=='__main__':
    # get class
    path1 = r'E:\\09-Others\\Try\\T\\7.bmp'
    path2 = r'E:\\09-Others\\Try\\N\\7.jpg'
    F1 = get_feature(path1)
    F2 = get_feature(path2)
    F1 = np.square(F1)
    F2 = np.square(F2)
    d = np.subtract(F1, F2)
    d = np.square(d)
    plt.imshow(d, 'gray'), plt.show()
