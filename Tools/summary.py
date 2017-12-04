import tensorflow as tf


class Summary:
    def __init__(self, session, dir_summary):
        self.__sess = session
        self.__vars = {}
        self.__ops = None
        self.__dir = dir_summary
        self.__writer = tf.summary.FileWriter(dir_summary, session.graph)
        self.__steps = 0

    def add_variable(self, var, name="name"):
        tf.summary.scalar(name, var)
        assert name not in self.__vars, "Already has " + name
        self.__vars[name] = var

    def build(self):
        self.__ops = tf.summary.merge_all()

    def run(self, feed_dict):
        feed_dict_final = {}
        for key, val in feed_dict.items():
            feed_dict_final[self.__vars[key]] = val
        str_summary = self.__sess.run(self.__ops, feed_dict_final)
        self.__writer.add_summary(str_summary, self.__steps)
        self.__writer.flush()
        self.__steps += 1

    def write_variables(self, flags):
        with open(self.__dir + '/vars.txt', 'w') as file_out:
            for key, val in flags.__dict__['__flags'].items():
                print(key, val, file=file_out)
