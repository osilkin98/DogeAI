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


# we should create a dataset of 385 doge images
def create_dataset(directory1, directory2, training=True):
    """This creates a dataset and returns a 4-D numpy array of imagesand a 1-D array of their respective labels,
        respectively.the training data to all data size should be 6/7 or ~85.71%, with the other test data
        comprising 14.29% of the total data size, or 1/7 to test the CNN"""

    dir1, dir2 = os.listdir(directory1), os.listdir(directory2)
    dir1_size, dir2_size = len(dir1), len(dir2)
    print("{} size: {}\n{} size: {}\nTotal: {}".format(directory1, dir1_size, directory2, dir2_size, dir1_size+dir2_size))

    dir1_chance = dir1_size / (dir1_size + dir2_size)
    print("Chance to pick item from {}: {}\nChance to pick item from {}: {}".format(
        directory1, dir1_chance,
        directory2, 1 - dir1_chance
        )
    )


create_dataset('/home/oleg/Pictures/classification_data/doge/', '/home/oleg/Pictures/classification_data/not-doge/')
# numpy_array = prepare_data_from_directory('/home/oleg/Pictures/classification_data/doge/')
# print(numpy_array.shape)