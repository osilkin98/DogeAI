from PIL import Image
import os
import wget

# Should be a class for a temporary image obtained from a URL
# Which is downloaded in the tmp/ directory and deleted once it's no longer in reference.
# This is really just a wrapper for the PIL.Image class

temp_directory = 'tmp/'


class temp_image:
    # Default constructor
    def __init__(self, outside_url=None):
        try:
            self.url = outside_url
            self.relative_path = temp_directory + "image{}.jpg".format(len(os.listdir(temp_directory)) + 1)
            wget.download(self.url, self.relative_path)
            self.image = Image.open(self.relative_path)
        except Exception as wget_exception: # this would likely happen in the case of a 40* error
            print(wget_exception.message)
        finally:
            if os.path.exists(self.relative_path):
                os.remove(self.relative_path)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.remove(self.relative_path)

    def __del__(self):
        os.remove(self.relative_path)