import tflearn
import numpy as np
import tensorflow as tf


def _builder(input_layer):
  net = tflearn.conv_2d(input_layer, 64, 3, strides=1, activation='relu')
  net = tflearn.max_pool_2d(net, 2, strides=2)
  net = tflearn.local_response_normalization(net)
  net = tflearn.conv_2d(net, 128, 3, activation='relu')
  net = tflearn.max_pool_2d(net, 2, strides=2)
  net = tflearn.local_response_normalization(net)
  net = tflearn.conv_2d(net, 256, 3, activation='relu')
  net = tflearn.conv_2d(net, 512, 3, activation='relu')
  net = tflearn.max_pool_2d(net, 2, strides=1)
  net = tflearn.local_response_normalization(net)

  return net


def build_model(image_shape, n_classmap, n_labels, net_builder=_builder):

  img_prep = tflearn.data_preprocessing.ImagePreprocessing()
  img_prep.add_featurewise_zero_center()
  img_prep.add_featurewise_stdnorm()

  img_aug = tflearn.data_augmentation.ImageAugmentation()
  img_aug.add_random_flip_leftright()
  img_aug.add_random_rotation(max_angle=25.)

  label_layer = tf.placeholder(tf.float32, shape=[None, n_labels])
  input_layer = tflearn.input_data(
      shape=[None, image_shape[0], image_shape[1], image_shape[2]],
      data_preprocessing=img_prep, data_augmentation=img_aug)

  net = net_builder(input_layer)
  conv_layer = tflearn.conv_2d(
      net, filter_size=2, strides=1, nb_filter=n_classmap, padding='valid')

  gap_layer = tf.reduce_mean(conv_layer, [1, 2])
  net = tflearn.fully_connected(
      gap_layer, n_labels, bias=False, activation='linear')
  gap_w = net.W
  tflearn.helpers.add_weights_regularizer(gap_w, loss="L1")

  with tf.variable_scope('classmap'):
    label_w = tf.gather(tf.transpose(gap_w), tf.argmax(label_layer, axis=1))
    label_w = tf.reshape(label_w, [-1, n_classmap, 1])

    conv_resize = tf.reshape(
        tf.image.resize_bilinear(
            conv_layer, [image_shape[0], image_shape[1]]),
        [-1, image_shape[0] * image_shape[1], n_classmap])

    classmap_layer = tf.reshape(
        tf.matmul(conv_resize, label_w), [-1, image_shape[0], image_shape[1]])

  optimizer = tflearn.optimizers.Momentum(
      learning_rate=0.01, momentum=0.9, lr_decay=0.99,
      decay_step=100, staircase=False, use_locking=False, name='Momentum')

  output_layer = tflearn.regression(
      net, label_layer, optimizer=optimizer,
      loss='softmax_categorical_crossentropy')
  model = tflearn.DNN(output_layer, tensorboard_verbose=2)

  return model, input_layer, label_layer, \
      output_layer, conv_layer, classmap_layer


def infer_classmap_f_generator(session, n_labels, input_layer, label_layer,
                               output_layer, conv_layer, classmap_layer):
  def infer_classmap(images, labels):
    conv, output = session.run(
        [conv_layer, output_layer], feed_dict={input_layer: images})
    prediction_labels = tf.cast(
        tf.one_hot(tf.argmax(output, axis=1), depth=n_labels),
        tf.int32).eval(session=session)

    classmap_values = session.run(classmap_layer, feed_dict={
        conv_layer: conv,
        label_layer: prediction_labels})
    classmap_answers = session.run(classmap_layer, feed_dict={
        conv_layer: conv,
        label_layer: tf.one_hot(
            tf.argmax(labels, axis=1), depth=n_labels).eval(session=session)})
    classmap_vis = np.array(list(map(
        lambda x: ((x - x.min()) / (x.max() - x.min())),
        classmap_answers)))

    return classmap_values, classmap_answers, classmap_vis

  return infer_classmap
