import cv2
import os
import numpy as np
doge_directory = '/home/oleg/Pictures/classification_data/doge/'
doge_list = os.listdir(doge_directory)
image_data_list = []
for x in doge_list:
    # print(x)
    image_data_list.append(cv2.imread("{}{}".format(doge_directory, x)))
print(image_data_list)
numpy_array = np.array(image_data_list)
print("numpy_array type: {}".format(type(numpy_array)))
print("doge image list type: {}".format(type(image_data_list)))
print(numpy_array.shape)
# image = cv2.imread('/home/oleg/Pictures/doge-jpeg-jpg/28iw3x.jpg')
# print(type(image))
# print(image)


# Takes a directory
def prepare_data_from_directory(directory):
    if directory[-1] != '/':
        raise Exception("Invalid Directory")
    try:
        directory_listing = os.listdir(directory)
        to_fill = []
        for x in directory_listing:
            # Appends a cv2 interpretation of a JPG image
            # CV2 representation is in a native numpy array type with dtype=uint8
            to_fill.append(cv2.imread("{}{}".format(directory, x)))
    except FileNotFoundError as fnf:
        print("[prepare_data_from_directory]: {} could not be found".format(fnf.filename))
    except NotADirectoryError as ndr:
        print("[prepare_data_from_directory]: invalid directory: {}".format(ndr.filename))
    except IsADirectoryError as idr:
        print("[prepare_data_from_directory]: is a directory, not file: {}".format(idr.filename))
    finally:
        return np.array(to_fill)