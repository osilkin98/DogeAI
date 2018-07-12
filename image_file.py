from PIL import Image
import os
import wget
import cv2
import numpy as np
# Should be a class for a temporary image obtained from a URL
# Which is downloaded in the tmp/ directory and deleted once it's no longer in reference.
# This is really just a wrapper for the PIL.Image class

temp_directory = 'tmp/'


class TempImage(object):
    # Default constructor
    def __init__(self, outside_url=None):
        try:
            self.url = outside_url
            self.relative_path = temp_directory + "image{}.jpg".format(len(os.listdir(temp_directory)) + 1)
            wget.download(self.url, self.relative_path)
            self.image = cv2.imread(self.relative_path)
            # If the image is incorrectly sized
            if self.image.shape[0] != 96 or self.image.shape[1] != 96:
                self.image = cv2.resize(self.image, (96, 96))
            self.image = np.true_divide(self.image.astype(np.float32), 255)
        except Exception as wget_exception: # this would likely happen in the case of a 40* error
            print(wget_exception)
            if os.path.exists(self.relative_path):
                os.remove(self.relative_path)

    def __del__(self):
        if os.path.exists(self.relative_path):
            os.remove(self.relative_path)

    def get_numpy(self):
        img_array = [self.image]
        return np.array(img_array)