import cv2
import os
import numpy as np
import random

# Takes a directory
def prepare_data_from_directory(directory):
    if directory[-1] != '/':
        raise Exception("Invalid Directory")

    to_fill = []
    try:
        directory_listing = os.listdir(directory)
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


# This Function should accept parameters as directory listings that end in '/'
def create_dataset(directory1, directory2):
    """This creates a dataset and returns a 4-D numpy array of imagesand a 1-D array of their respective labels,
        respectively.the training data to all data size should be 6/7 or ~85.71%, with the other test data
        comprising 14.29% of the total data size, or 1/7 to test the CNN"""

    if directory1[-1] != '/' or directory2[-1] != '/':
        raise Exception("Invalid Directory Arguments")

    # For the labels, 1 == doge, 0 == not doge
    img_array, labels = [], []

    try:
        # since os.listdir() returns a listing of all the files but not the absolute path, for reading the
        # images in with cv2 we'll have to simply appened the directory name
        dir1, dir2 = os.listdir(directory1), os.listdir(directory2)
        files = [dir1, dir2]
        total_size = len(files[0]) + len(files[1])
        # Move all the files from the directory listing into cv2 arrays, and then remove them from the program's memory
        while total_size > 0:

            # use p <= doge_size / total_size for a discrete random variable p to ensure a rolling even spread of 50/50
            if len(files[1]) == 0 or (random.random() <= float(len(files[0])) / total_size and len(files[0]) != 0):

                # pick a random doge file from our set of files
                random_doge = files[0][random.randint(0, len(files[0]) - 1)]

                # if the file selected is a jpeg
                if (random_doge[len(random_doge)-4:len(random_doge)].lower() == ".jpg") or \
                        (random_doge[len(random_doge)-5:len(random_doge)].lower() == ".jpeg"):
                    try:

                        # Read it in with cv2
                        doge_image = cv2.imread("{}{}".format(directory1, random_doge))

                        # If its dimensions are something other than 96x96, we just reshape it
                        if doge_image.shape[0] != 96 or doge_image.shape[1] != 96:
                            doge_image = cv2.resize(doge_image, (96, 96))

                        # Format the image type as a 32-bit float and append it to the array
                        img_array.append(np.true_divide(doge_image.astype(np.float32), 255))

                        # append the doge label
                        labels.append(1)

                    # if something went wrong, we print the base exception
                    except Exception as e:
                        print("{}".format(e))
                    finally:

                        # remove the doge_file from the program's memory
                        files[0].remove(random_doge)

                # if the files are unrecognized
                else:
                    # remove it from the program's memory
                    files[0].remove(random_doge)

                    # decrement the total size
                    total_size -= 1
                    continue
            # otherwise we're looking at non-doge files
            else:
                # pick a not doge at random
                ''' Essentially at this point all the code is virtually the same as above 
                    so there's no need to really annotate anything and since I'm lazy I never moved 
                    these very functional routines to another function.  ¯\_(ツ)_/¯ 
                '''
                random_notdoge = files[1][random.randint(0, len(files[1]) - 1)]
                if (random_notdoge[len(random_notdoge)-4:len(random_notdoge)].lower() == ".jpg") or \
                    (random_notdoge[len(random_notdoge)-5:len(random_notdoge)].lower() == ".jpeg"):
                    try:
                    # read image in with cv2
                        random_image = cv2.imread("{}{}".format(directory2, random_notdoge))
                        if random_image.shape[0] != 96 or random_image.shape[1] != 96:
                            random_image = cv2.resize(random_image, (96, 96))
                        img_array.append(np.true_divide(random_image.astype(np.float32), 255))
                        labels.append(0)
                    except Exception as e:
                        print("{}".format(e))
                    finally:
                        files[1].remove(random_notdoge)
                # if the files are not recognized then we just continue
                else:
                    files[1].remove(random_notdoge)
                    total_size -= 1
                    continue
            total_size -= 1
    except FileNotFoundError as fnf:
        print("[prepare_data_from_directory]: {} could not be found".format(fnf.filename))
    except NotADirectoryError as ndr:
        print("[prepare_data_from_directory]: invalid directory: {}".format(ndr.filename))
    except IsADirectoryError as idr:
        print("[prepare_data_from_directory]: is a directory, not file: {}".format(idr.filename))
    except Exception as e:
        print(e)
    finally:
        return np.array(img_array, dtype=np.float32), np.array(labels, dtype=np.int32)


def get_numpy_image_array(image_path):
    img_array = []
    try:
        image = cv2.imread(image_path)
        # print(image)
        if image.shape[0] != 96 or image.shape[1] != 96:
            print("{} are incorrect dimensions, resizing...".format(image.shape))
            image = cv2.resize(image, (96, 96))
        print("image: ")
        img_array.append(np.true_divide(image.astype(np.float32), 255))
        print("Image Array:")
    except FileNotFoundError as e:
        print("{} does not exist".format(e.filename))
    except NotADirectoryError as e:
        print("{} is not a directory, strerror: {}".format(e.filename, e.strerror))
    except Exception as e:
        print("Exception Called: {}".format(e))
    finally:
        return np.array(img_array)