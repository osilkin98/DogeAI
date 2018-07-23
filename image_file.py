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
            if not os.path.exists("{}/{}".format(os.getcwd(),temp_directory)):
                os.mkdir("{}/{}".format(os.getcwd(), temp_directory))

            # tries to get the url from the outside world
            self.url = outside_url

            # sets the relative path to what will become the image object
            # this is really where things might go wrong because the 'tmp/' directory may not always be
            # present on someone's configuration. Though in practice it might be worthwhile to just add the
            # tmp directory to vcs, it's not a good idea because it would be an empty directory.
            try:
                self.relative_path = temp_directory + "image{}.jpg".format(len(os.listdir(temp_directory)) + 1)
                self.absolute_path = "{}/{}".format(os.getcwd(), self.relative_path)
            except Exception as ae:
                print("Attribute error was raised: {}".format(ae))

            finally:

                # downloads the image to the path that we specified
                wget.download(self.url, self.relative_path)

            # reads the image in from the cv2 library using our relative path that we defined above
            self.image = cv2.imread(self.relative_path)

            # If the image is incorrectly sized, we just change its size to
            if self.image.shape[0] != 96 or self.image.shape[1] != 96:
                self.image = cv2.resize(self.image, (96, 96))

            # at this point we just change the data format to be a 32 bit floating point number,
            # and divide through by 255.0 in order to get the values as a percentage in order to make it easier
            # for TensorFlow to interpret our image
            self.image = np.true_divide(self.image.astype(np.float32), 255)

        except Exception as wget_exception: # this would likely happen in the case of a 40* error
            print(wget_exception)
            if os.path.exists(self.relative_path):
                os.remove(self.relative_path)

    # to remove the object from the tmp/ directory before being completely called off the stack
    def __del__(self):
        if os.path.exists(self.relative_path):
            print("Removing file from {}".format(self.relative_path))
            os.remove(self.relative_path)

    def get_numpy(self):
        img_array = [self.image]
        return np.array(img_array)