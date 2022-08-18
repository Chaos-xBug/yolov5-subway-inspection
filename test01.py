import cv2
import os
path_base = r'E:\04-Data\02-robot\04-Xingqiao0512\3-CollectedData\cutimages\template\Bolt\Pic_2022_05_12_145412_2_Bolt0.jpg'
img = cv2.imread(r'E:\04-Data\02-robot\05-Xingqiao0708\test1-detectresults\T1B1.jpg')
cv2.imwrite(path_base,img)
