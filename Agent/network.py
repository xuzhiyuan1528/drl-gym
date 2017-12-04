import tensorflow as tf
import tflearn


def build_flat_net(dim_s, dim_a):
    inputs = tf.placeholder(tf.float32, shape=[None] + dim_s)
    net = tf.contrib.layers.flatten(inputs)

    init_weight = tflearn.initializations.truncated_normal(stddev=0.01)

    net = tflearn.fully_connected(net, 300, activation='leakyRelu', weights_init=init_weight)
    net = tflearn.fully_connected(net, 200, activation='leakyRelu', weights_init=init_weight)
    net = tflearn.fully_connected(net, 100, activation='leakyRelu', weights_init=init_weight)

    q_values = tflearn.fully_connected(net, dim_a)

    return inputs, q_values

def build_cnn_net(dim_s, dim_a):
    inputs = tf.placeholder(tf.float32, shape=[None] + dim_s)

    net = tflearn.conv_2d(inputs, 32, 8, strides=4, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.batch_normalization(net)

    net = tflearn.conv_2d(net, 64, 4, strides=2, activation='relu')
    net = tflearn.max_pool_2d(net, 2)
    net = tflearn.batch_normalization(net)

    net = tflearn.fully_connected(net, 128, activation='relu')

    net = tflearn.fully_connected(net, 256, activation='relu')

    q_values = tflearn.fully_connected(net, dim_a)

    return inputs, q_values
