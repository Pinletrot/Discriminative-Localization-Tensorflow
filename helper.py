import tflearn
import tensorflow as tf


def build_network_core(input_layer):
  net = tflearn.conv_2d(input_layer, 64, 11, strides=4, activation='relu')
  net = tflearn.max_pool_2d(net, 3, strides=2)
  net = tflearn.local_response_normalization(net)
  net = tflearn.conv_2d(net, 128, 5, activation='relu')
  net = tflearn.max_pool_2d(net, 3, strides=2)
  net = tflearn.local_response_normalization(net)
  net = tflearn.conv_2d(net, 256, 3, activation='relu')
  net = tflearn.conv_2d(net, 256, 3, activation='relu')
  net = tflearn.conv_2d(net, 512, 3, activation='relu')
  net = tflearn.max_pool_2d(net, 3, strides=2)
  net = tflearn.local_response_normalization(net)

  return net


def build_model(session, image_shape, n_classmap, n_labels,
                loss_func='categorical_crossentropy',
                learning_rate=0.003, network_build_func=build_network_core):
  label_layer = tf.placeholder(tf.float32, shape=[None, n_labels])
  input_layer = tflearn.input_data(
      shape=[None, image_shape[0], image_shape[1], 3])

  net = network_build_func(input_layer)
  conv_layer = tflearn.conv_2d(net, n_classmap, 3, activation='relu')

  try:
    with tf.variable_scope('GAP'):
      gap_w = tf.get_variable(
          'W', shape=[n_classmap, n_labels],
          initializer=tf.random_normal_initializer(0., 0.01))
  except ValueError:
    with tf.variable_scope('GAP', reuse=True):
      gap_w = tf.get_variable('W')

  gap = tf.reduce_mean(conv_layer, axis=[1, 2])
  net = tf.matmul(gap, gap_w)
  net = tflearn.fully_connected(net, n_labels, activation='softmax')
  output_layer = tflearn.regression(
      net, label_layer, loss=loss_func, learning_rate=learning_rate)

  model = tflearn.DNN(output_layer, tensorboard_verbose=2, session=session)

  with tf.variable_scope('classmap'):
    label_w = tf.gather(tf.transpose(gap_w), tf.argmax(label_layer, axis=1))
    label_w = tf.reshape(label_w, [-1, n_classmap, 1])

    conv_resize = tf.reshape(
        tf.image.resize_bilinear(conv_layer, [image_shape[0], image_shape[1]]),
        [-1, image_shape[0] * image_shape[1], n_classmap])
    classmap_layer = tf.reshape(
        tf.matmul(conv_resize, label_w), [-1, image_shape[0], image_shape[1]])

  session.run(tf.global_variables_initializer())
  return model, input_layer, label_layer, \
      output_layer, conv_layer, classmap_layer


def infer_classmap_f_generator(session, n_labels, input_layer, label_layer,
                               output_layer, conv_layer, classmap_layer):
  def infer_classmap(images, labels):
    conv, output = session.run([conv_layer, output_layer],
                               feed_dict={input_layer: images})
    prediction_labels = tf.cast(
        tf.one_hot(tf.argmax(output, axis=1), depth=n_labels),
        tf.int32).eval(session=session)
    classmap_values = session.run(classmap_layer, feed_dict={
        conv_layer: conv,
        label_layer: prediction_labels
    })
    classmap_answers = session.run(classmap_layer, feed_dict={
        conv_layer: conv,
        label_layer: labels
    })
    classmap_vis = list(map(lambda x: ((x - x.min()) / (x.max() - x.min())),
                            classmap_answers))

    return classmap_values, classmap_answers, classmap_vis

  return infer_classmap


def plot_classmap(images, labels, infer_classmap_f, plot=False):
  import matplotlib.pyplot as plt

  values, answers, vis = infer_classmap_f(images, labels)
  if plot:
    for i in range(len(images)):
      f, ax = plt.subplots(1, 3)
      ax[0].imshow(images[i])
      ax[0].imshow(vis[i], cmap=plt.cm.jet, alpha=0.5, interpolation='nearest')
      ax[1].imshow(values[i])
      ax[2].imshow(answers[i])

  return values, answers, vis
