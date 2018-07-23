from shutil import copyfile
import os
import random
import time

base_data_dirs = {
    "doge": "/home/oleg/Pictures/classification_data/doge/",
    "not-doge": "/home/oleg/Pictures/classification_data/not-doge/"
}

training_data_dirs = {
    "doge": "/home/oleg/Pictures/classification_data/training/doge/",
    "not-doge": "/home/oleg/Pictures/classification_data/training/not-doge/"
}

testing_data_dirs = {
    "doge": "/home/oleg/Pictures/classification_data/testing/doge/",
    "not-doge": "/home/oleg/Pictures/classification_data/testing/not-doge/"
}

doge_list = os.listdir(base_data_dirs["doge"])
notdoge_list = os.listdir(base_data_dirs["not-doge"])

file_list = [doge_list, notdoge_list]

list_count = len(file_list[0]) + len(file_list[1])

print("Total: {}; {}/7= {}".format(list_count, list_count, float(list_count) / 7.0))

print(float(len(file_list[0])) / list_count)

random.seed(time.clock())

while (list_count > 110):  # Moving files into the training data directory
    # if the random value generated is less than or equal to the amount of doge files
    if (len(file_list[1]) == 0 or (
            random.random() <= float((len(file_list[0]))) / list_count and len(file_list[0]) != 0)):
        # pick a random doge
        random_doge = file_list[0][random.randint(0, len(file_list[0]) - 1)]
        print("Copying {} from {}{} to {}{}".format(random_doge, base_data_dirs['doge'], random_doge,
                                                    training_data_dirs['doge'], random_doge))
        copyfile("{}{}".format(base_data_dirs['doge'], random_doge),
                 "{}{}".format(training_data_dirs['doge'], random_doge))
        print("Done, removing {} from file list".format(random_doge))
        file_list[0].remove(random_doge)
    else:
        random_notdoge = file_list[1][random.randint(0, len(file_list[1]) - 1)]
        print("Copying {} from {}{} to {}{}".format(random_notdoge, base_data_dirs['not-doge'], random_notdoge,
                                                    training_data_dirs['not-doge'], random_notdoge))
        copyfile("{}{}".format(base_data_dirs['not-doge'], random_notdoge),
                 "{}{}".format(training_data_dirs['not-doge'], random_notdoge))
        print("Done, removing {} from file_list".format(random_notdoge))
        file_list[1].remove(random_notdoge)
    list_count -= 1

while (list_count != 0):
    if (len(file_list[1]) == 0 or (
            random.random() <= float((len(file_list[0]))) / list_count and len(file_list[0]) != 0)):
        # pick a random doge
        random_doge = file_list[0][random.randint(0, len(file_list[0]) - 1)]
        print("Copying {} from {}{} to {}{}".format(random_doge, base_data_dirs['doge'], random_doge,
                                                    testing_data_dirs['doge'], random_doge))
        copyfile("{}{}".format(base_data_dirs['doge'], random_doge),
                 "{}{}".format(testing_data_dirs['doge'], random_doge))
        print("Done, removing {} from file list".format(random_doge))
        file_list[0].remove(random_doge)
    else:
        random_notdoge = file_list[1][random.randint(0, len(file_list[1]) - 1)]
        print("Copying {} from {}{} to {}{}".format(random_notdoge, base_data_dirs['not-doge'], random_notdoge,
                                                    testing_data_dirs['not-doge'], random_notdoge))
        copyfile("{}{}".format(base_data_dirs['not-doge'], random_notdoge),
                 "{}{}".format(testing_data_dirs['not-doge'], random_notdoge))
        print("Done, removing {} from file_list".format(random_notdoge))
        file_list[1].remove(random_notdoge)
    list_count -= 1
