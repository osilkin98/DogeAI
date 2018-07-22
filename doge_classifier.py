from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow as tf
import numpy as np
import dogedata
import cv2

tf.logging.set_verbosity(tf.logging.ERROR)


def doge_convolution(features, labels, mode):
    # first input layer
    input_layer = tf.reshape(features["x"], [-1, 96, 96, 3])

    # first convolutionary layer: 96x96x12 = 110592
    convolutionary_layer_1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=12,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )
    # reduces 96x96x12 -> 48x48x12 # conversion rate = 2
    pooling_layer_1 = tf.layers.max_pooling2d(
        inputs=convolutionary_layer_1,
        pool_size=[2, 2],
        strides=2
    )
    # second convolutional layer: 48x48x24 = 55,296
    convolutionary_layer_2 = tf.layers.conv2d(
        inputs=pooling_layer_1,
        filters=24,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu
    )

    # second pooling layer: 48x48x24 -> 24x24x24 # conversion rate = 4
    pooling_layer_2 = tf.layers.max_pooling2d(
        inputs=convolutionary_layer_2,
        pool_size=[2, 2],
        strides=2
    )

    # third convolutional layer: 24x24x24 = 13,824
    convolutionary_layer_3 = tf.layers.conv2d(
        inputs=pooling_layer_2,
        filters=24,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    # third pooling layer: 24x24x24 -> 12x12x48 # conversion rate = 2
    pooling_layer_3 = tf.layers.max_pooling2d(
        inputs=convolutionary_layer_3,
        pool_size=[2, 2],
        strides=2
    )

    # fourth convolutional layer: 12x12x48 = 3,456
    convolutionary_layer_4 = tf.layers.conv2d(
        inputs=pooling_layer_3,
        filters=48,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu
    )

    # fourth pooling layer: 12x12x48 -> 6x6x48
    pooling_layer_4 = tf.layers.max_pooling2d(
        inputs=convolutionary_layer_4,
        pool_size=[2, 2],
        strides=2
    )
    # single layer with 1,728 neurons
    flatten_pooling_layer_2 = tf.reshape(pooling_layer_4, [-1, 6 * 6 * 48])

    dense = tf.layers.dense(inputs=flatten_pooling_layer_2, units=1024, activation=tf.nn.relu)

    dropout = tf.layers.dropout(inputs=dense, rate=0.1, training=mode == tf.estimator.ModeKeys.TRAIN)

    # possible outcomes: doge or not doge
    logits = tf.layers.dense(inputs=dropout, units=2)

    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        "probabilities": tf.nn.sigmoid(logits, name="sigmoid_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate loss for training and evaluation modes
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    eval_ops_metrics = {
        "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])
    }

    # Return in evaluation mode
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_ops_metrics)



def main(unused_argv):
    # train_data is a int32 training data type
    train_data, train_labels = dogedata.create_dataset(
        "/home/oleg/Pictures/Doge/",
        "/home/oleg/Pictures/classification_data/training/not-doge/")
    eval_data, eval_labels = dogedata.create_dataset(
        "/home/oleg/Pictures/classification_data/testing/doge/",
        "/home/oleg/Pictures/classification_data/testing/not-doge/"
    )
    # Create an Estimator object which links the doge_convolution function as the training model
    # And uses tmp/ to store the model results
    classifier = tf.estimator.Estimator(model_fn=doge_convolution,
                                        model_dir='trained_doge/',
                                        config=tf.estimator.RunConfig(log_step_count_steps=250))

    tensors_to_log = {"probabilities": "sigmoid_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log,
                                              every_n_iter=500)

    epoch_count, data_size = 40, len(train_data)
    print(data_size)

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_data},
        y=train_labels,
        batch_size=20,
        num_epochs=epoch_count,
        shuffle=True
    )

    aster = 40

    for i in range(1, epoch_count+1):
        print("{} ITERATION {} {}".format('*'*aster, i, '*'*aster))
        classifier.train(input_fn=train_input_fn,
                         hooks=[logging_hook],
                         steps=data_size*epoch_count)

    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_data},
        y=eval_labels,
        num_epochs=1,
        shuffle=False
    )
    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)



if __name__ == "__main__":
  tf.app.run()
