import tensorflow as tf
import tflearn

from Agent.network import build_flat_net, build_cnn_net


class DDQNAgent():
    def __init__(self, session, dim_state, dim_action, learning_rate, tau, net_name='cnn'):
        self.__sess = session
        self.__dim_s = dim_state
        self.__dim_a = dim_action
        self.__lr = learning_rate
        self.__tau = tau

        if net_name == 'flat':
            self.__inputs, self.__out = build_flat_net(dim_state, dim_action)
            self.__paras = tf.trainable_variables()
            self.__target_inputs, self.__target_out = build_flat_net(dim_state, dim_action)
        else:
            self.__inputs, self.__out = build_cnn_net(dim_state, dim_action)
            self.__paras = tf.trainable_variables()
            self.__target_inputs, self.__target_out = build_cnn_net(dim_state, dim_action)

        self.__target_paras = tf.trainable_variables()[(len(self.__paras)):]

        self.__ops_update_target = []
        for i in range(len(self.__target_paras)):
            val = tf.add(tf.multiply(self.__paras[i], self.__tau), tf.multiply(self.__target_paras[i], 1. - self.__tau))
            op = self.__target_paras[i].assign(val)
            self.__ops_update_target.append(op)

        self.__actions = tf.placeholder(tf.float32, [None, self.__dim_a])
        self.__y_values = tf.placeholder(tf.float32, [None])

        action_q_values = tf.reduce_sum(tf.multiply(self.__out, self.__actions), reduction_indices=1)

        self.loss = tflearn.mean_square(self.__y_values, action_q_values)
        self.optimize = tf.train.AdamOptimizer(self.__lr).minimize(self.loss)

    def train(self, inputs, action, y_values):
        return self.__sess.run([self.optimize, self.loss], feed_dict={
            self.__inputs: inputs,
            self.__actions: action,
            self.__y_values: y_values
        })

    def predict(self, inputs):
        return self.__sess.run(self.__out, feed_dict={
            self.__inputs: inputs,
        })

    def predict_target(self, inputs):
        return self.__sess.run(self.__target_out, feed_dict={
            self.__target_inputs: inputs,
        })

    def update_target_paras(self):
        self.__sess.run(self.__ops_update_target)
