import tflearn
import numpy as np
import tensorflow as tf

from helper import build_model, infer_classmap_f_generator, plot_classmap


def network_builder(input_layer):
  net = tflearn.conv_2d(input_layer, 32, 3, strides=2, activation='relu')
  net = tflearn.max_pool_2d(net, 3, strides=2)
  net = tflearn.local_response_normalization(net)
  net = tflearn.conv_2d(net, 32, 3, activation='relu')
  net = tflearn.max_pool_2d(net, 3, strides=2)
  net = tflearn.local_response_normalization(net)
  net = tflearn.conv_2d(net, 64, 3, activation='relu')
  net = tflearn.conv_2d(net, 128, 3, activation='relu')
  net = tflearn.conv_2d(net, 32, 3, activation='relu')
  net = tflearn.max_pool_2d(net, 3, strides=2)

  return net


sess = tf.InteractiveSession()

# load training data
X, Y = tflearn.datasets.mnist.load_data(data_dir='data/mnist', one_hot=True)

# build model
model, input_layer, label_layer, output_layer, conv_layer, classmap_layer \
    = build_model(session=sess, image_shape=[28, 28], n_labels=10,
                  n_classmap=64, network_build_func=network_builder)

# construct infer function
infer_classmap = infer_classmap_f_generator(
    sess, 10, input_layer, label_layer,
    output_layer, conv_layer, classmap_layer)

# train model
model.fit(
    X, Y, n_epoch=1, validation_set=0.1,
    batch_size=96, shuffle=True, show_metric=True)

# plot classmap
values, answers, vis = plot_classmap(X[:2], Y[:2], infer_classmap, plot=True)
