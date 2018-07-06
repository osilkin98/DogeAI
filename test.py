import tensorflow as tf
import numpy as np
from PIL import Image
import dircache

directory = '/home/oleg/Pictures/doge-jpeg-jpg/'
old_file_list = dircache.opendir(directory)
file_list = []
for filename in old_file_list:
    file_list.append(directory + filename)

print(file_list)


print("Using TensorFlow version {}".format(tf.VERSION))

filename_queue = tf.train.string_input_producer(file_list) #  list of files to read

reader = tf.WholeFileReader()
key, value = reader.read(filename_queue)

my_img = tf.image.decode_jpeg(value) # use png or jpg decoder based on your files.

init_op = tf.global_variables_initializer()

def doge_convolution(features, labels, mode):
    # first input layer
    input_layer = tf.reshape(features["x"], [-1, 100, 100, 3])


    # second input layer
    convolutionary_layer_1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=12,
        kernel_size=[30, 30],
        padding="same",
        activation=tf.nn.relu
    )

    pooling_layer_1 = tf.layers.max_pooling2d(
        inputs=convolutionary_layer_1,
        pool_size=[2, 2],
        strides=2
    )

    convolutionary_layer_2 = tf.layers.conv2d(
        inputs=pooling_layer_1,
        filters=24,
        kernel_size=[30, 30],
        padding="same",
        activation=tf.nn.relu
    )

    pooling_layer_2 = tf.layers.average_pooling2d(
        inputs=convolutionary_layer_2,
        pool_size=[2, 2],
        strides=2
    )



'''
with tf.Session() as sess:
    sess.run(init_op)

    # Start populating the filename queue.

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)
    f = 0
    for i in range(len(file_list)): #length of your filename list
        try:
            image = my_img.eval() #here is your image Tensor :)
            f += 1
        except Exception as e:
            print(e.message)
            print("exception raised with file {} at index: {}".format(file_list[i], str(i)))
        finally:
            pass

    print("i = {}, f = {}".format(i, f))
    print(image.shape)
    # Image.fromarray(np.asarray(image)).show()

    coord.request_stop()
    coord.join(threads)
'''