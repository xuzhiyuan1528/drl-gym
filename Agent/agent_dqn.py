import tensorflow as tf
import tflearn

from Agent.network import build_flat_net, build_cnn_net


class DQNAgent():
    def __init__(self, session, dim_state, dim_action, learning_rate, net_name='cnn'):
        self.__sess = session
        self.__dim_s = dim_state
        self.__dim_a = dim_action
        self.__lr = learning_rate

        if net_name == 'flat':
            self.__inputs, self.__out = build_flat_net(dim_state, dim_action)
        else:
            self.__inputs, self.__out = build_cnn_net(dim_state, dim_action)

        self.__actions = tf.placeholder(tf.float32, [None, self.__dim_a])
        self.__y_values = tf.placeholder(tf.float32, [None])

        self.__action_q_values = tf.reduce_sum(tf.multiply(self.__out, self.__actions), reduction_indices=1)

        self.loss = tflearn.mean_square(self.__y_values, self.__action_q_values)
        self.optimize = tf.train.AdamOptimizer(self.__lr).minimize(self.loss)

    def train(self, inputs, action, y_values):
        return self.__sess.run([self.__action_q_values, self.loss], feed_dict={
            self.__inputs: inputs,
            self.__actions: action,
            self.__y_values: y_values
        })

    def predict(self, inputs):
        return self.__sess.run(self.__out, feed_dict={
            self.__inputs: inputs,
        })