from __future__ import print_function, absolute_import, division

import tflearn
import tensorflow as tf


class Detector:

  def __init__(self, n_labels, sess,
               is_preprocess=False, batch_size=32,
               loss='categorical_crossentropy',
               model_path='/tmp/tflearn_models/'):
    self.n_labels = n_labels
    self.sess = sess
    self.loss = loss
    self.batch_size = batch_size
    self.is_preprocess = is_preprocess
    self.run_id = 'Detector-%s-%s' % (n_labels, loss)

    self.labels = tf.placeholder(
        tf.float32,
        shape=[None, self.n_labels])
    self.input, self.output, self.classmap, self.gap, self.conv, \
        self.pools = self._build_network()

    self.model = tflearn.DNN(
        self.output, tensorboard_verbose=2,
        session=sess, checkpoint_path=model_path
    )
    try:
      self.model.load(model_path)
    except Exception:
      sess.run(tf.global_variables_initializer())

  def _build_network(self):
    if self.is_preprocess:
      # Real-time data preprocessing
      img_prep = tflearn.data_preprocessing.ImagePreprocessing()
      img_prep.add_featurewise_zero_center()
      img_prep.add_featurewise_stdnorm()

      # Real-time data augmentation
      img_aug = tflearn.data_augmentation.ImageAugmentation()
      img_aug.add_random_flip_leftright()
      img_aug.add_random_rotation(max_angle=25.)

      net = images = tflearn.input_data(
          shape=[None, 224, 224, 3],
          data_preprocessing=img_prep,
          data_augmentation=img_aug)
    else:
      net = images = tflearn.input_data(shape=[None, 224, 224, 3])

    net = tflearn.conv_2d(net, 32, 3, activation='relu')
    net = pool1 = tflearn.max_pool_2d(net, 2)

    net = tflearn.conv_2d(net, 64, 3, activation='relu')
    net = tflearn.conv_2d(net, 64, 3, activation='relu')
    net = pool2 = tflearn.max_pool_2d(net, [1, 2, 2, 1], strides=[1, 2, 2, 1])

    net = tflearn.conv_2d(net, 128, 3, activation='relu')
    net = tflearn.conv_2d(net, 128, 3, activation='relu')
    net = pool3 = tflearn.max_pool_2d(net, [1, 2, 2, 1], strides=[1, 2, 2, 1])

    net = tflearn.conv_2d(net, 256, 3, activation='relu')
    net = tflearn.conv_2d(net, 256, 3, activation='relu')
    net = pool4 = tflearn.max_pool_2d(net, [1, 2, 2, 1], strides=[1, 2, 2, 1])

    net = tflearn.conv_2d(net, 512, 3, activation='relu')
    net = tflearn.conv_2d(net, 512, 3, activation='relu')

    net = conv = tflearn.conv_2d(net, 1024, 3, activation='relu')

    with tf.variable_scope('GAP'):
      gap_w = tf.get_variable(
          'W', shape=[1024, self.n_labels],
          initializer=tf.random_normal_initializer(0., 0.01))

      net = gap = tf.reduce_mean(conv, axis=[1, 2])
      net = tf.matmul(net, gap_w)

    # net = tflearn.fully_connected(net, self.n_labels, activation='softmax')
    with tf.variable_scope('classmap'):
      label_w = tf.gather(tf.transpose(gap_w), tf.argmax(self.labels, axis=1))
      label_w = tf.reshape(label_w, [-1, 1024, 1])

      conv_resized = tf.image.resize_bilinear(conv, [224, 224])

      conv_resized = tf.reshape(conv_resized, [-1, 224 * 224, 1024])
      classmap = tf.matmul(conv_resized, label_w)
      classmap = tf.reshape(classmap, [-1, 224, 224])

    net = tflearn.regression(net, self.labels, loss=self.loss)

    return images, net, classmap, gap, conv, [pool1, pool2, pool3, pool4]

  def fit(self, X, y, batch_size=32, validation_set=0.1,
          n_epoch=1, shuffle=True, show_metric=True):

    self.model.fit(
        X, y, batch_size=batch_size, n_epoch=n_epoch, shuffle=shuffle,
        validation_set=validation_set, show_metric=show_metric,
        run_id=self.run_id)

  def predict(self, X):
    return self.model.predict(X)

  def predict_label(self, X):
    return self.model.predict_label(X)

  def infer_class_map(self, X, labels):
    conv, output = self.sess.run(
        [self.conv, self.output], feed_dict={
            self.input: X
        })

    prediction_labels = tf.cast(
        tf.one_hot(output.argmax(axis=1), depth=2),
        tf.int32).eval(session=self.sess)

    classmap_vals = self.sess.run(self.classmap, feed_dict={
        self.labels: prediction_labels, self.conv: conv})
    classmap_answers = self.sess.run(self.classmap, feed_dict={
        self.labels: labels, self.conv: conv})
    classmap_vis = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())),
                            classmap_answers))
    return classmap_vals, classmap_answers, classmap_vis
