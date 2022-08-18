from PIL import Image
import os
from classfier.siamese import Siamese

if __name__ == "__main__":
    model = Siamese()

    while True:
        image_1 = input('Input image_1 filename:')
        try:
            image_1 = Image.open(image_1)
        except:
            print('Image_1 Open Error! Try again!')
            continue
        print(type(image_1))

        image_2 = input('Input image_2 filename:')
        try:
            image_2 = Image.open(image_2)
        except:
            print('Image_2 Open Error! Try again!')
            continue
        probability = model.detect_image(image_1, image_2)
        print(probability)

    # path_base = r'C:\Users\Lenovo\PycharmProjects\yolov5-wjd\data\Line3\manques\1.bmp'
    # path_label_o = 'data\\Line3\\StandardDefautTree\\labels\\'
    # for filename in os.listdir(path_base):
    #     image_1 = Image.open(image_1)
    #     image_1 = Image.open(image_1)