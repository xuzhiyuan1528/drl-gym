import numpy as np


class Explorer:
    def __init__(self, epsilon_begin, epsilon_end, steps, flag):
        self.__ep_b = epsilon_begin
        self.__ep_e = epsilon_end
        self.__ep = epsilon_begin
        self.__steps = steps
        self.__replay_flag = flag

    @property
    def epsilon(self):
        return self.__ep

    def get_action(self, Q_value):
        Q_value = Q_value.tolist()
        action = np.zeros(len(Q_value))

        if np.random.random() <= self.__ep and not self.__replay_flag:
            action_index = np.random.randint(low=0, high=len(Q_value))
        else:
            action_index = np.argmax(Q_value)

        action[action_index] = 1

        if self.__ep > self.__ep_e and not self.__replay_flag:
            self.__ep -= (self.__ep_b - self.__ep_e) / self.__steps


        return action
