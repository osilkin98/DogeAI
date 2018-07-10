import cv2
import os
import numpy as np

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


numpy_array = prepare_data_from_directory('/home/oleg/Pictures/classification_data/doge/')
print(numpy_array.shape)