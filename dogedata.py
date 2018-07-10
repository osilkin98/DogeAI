import cv2
import os
import numpy as np
doge_directory = '/home/oleg/Pictures/doge-jpeg-jpg/'
doge_list = os.listdir(doge_directory)
image_data_list = []
for x in doge_list:
    print(x)
    image_data_list.append(cv2.imread("{}{}".format(doge_directory, x)))
print(image_data_list)
numpy_array = np.array(image_data_list)
print("numpy_array type: {}".format(type(numpy_array)))
print("doge image list type: {}".format(type(image_data_list)))
print(numpy_array)
# image = cv2.imread('/home/oleg/Pictures/doge-jpeg-jpg/28iw3x.jpg')
# print(type(image))
# print(image)