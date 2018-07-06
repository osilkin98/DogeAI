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
    input_layer = tf.reshape(features["x"], [-1, 96, 96, 3])

    # first convolutionary layer: 96x96x32 = 294,912
    convolutionary_layer_1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    # reduces 96x96x32 -> 48x48x32
    pooling_layer_1 = tf.layers.max_pooling2d(
        inputs=convolutionary_layer_1,
        pool_size=[2, 2],
        strides=2
    )
    # second convolutional layer: 48x48x64 = 147,456
    convolutionary_layer_2 = tf.layers.conv2d(
        inputs=pooling_layer_1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # second pooling layer: 48x48x64 -> 24x24x64
    pooling_layer_2 = tf.layers.max_pooling2d(
        inputs=convolutionary_layer_2,
        pool_size=[2, 2],
        strides=2
    )

    # third convolutional layer: 24x24x128 = 73,720
    convolutionary_layer_3 = tf.layers.conv2d(
        inputs=pooling_layer_2,
        filters=128,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    # third pooling layer: 24x24x128 -> 12x12x128
    pooling_layer_3 = tf.layers.max_pooling2d(
        inputs=convolutionary_layer_3,
        pool_size=[2, 2],
        strides=2
    )

    # fourth convolutional layer: 12x12x256 = 36,864
    convolutionary_layer_4 = tf.layers.conv2d(
        inputs=pooling_layer_3,
        filters=256,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    # fourth pooling layer: 12x12x256 -> 6x6x256
    pooling_layer_4 = tf.layers.max_pooling2d(
        inputs=convolutionary_layer_4,
        pool_size=[2, 2],
        strides=2
    )

    flatten_pooling_layer_2 = tf.reshape(pooling_layer_2, [-1, 6 * 6 * 256])

    dense = tf.layers.dense(inputs=flatten_pooling_layer_2, units=2048, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, training=mode == tf.estimator.ModeKeys.TRAIN)

    # possible outcomes: doge or not doge
    logits = tf.layers.dense(inputs=dropout, units=2)


    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)


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