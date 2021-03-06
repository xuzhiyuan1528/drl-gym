import datetime
import tensorflow as tf
import numpy as np
import os
import gym
import gym_ple

from Agent.agent_dqn import DQNAgent
from Tools.explorer import Explorer
from Tools.image import pre_process_image_catcher
from Tools.replaybuffer import ReplayBuffer
from Tools.summary import Summary
from flag_catcher import FLAGS

time_stamp = str(datetime.datetime.now().strftime('%y-%m-%d-%H-%M-%S'))

STATE_FRAMES = getattr(FLAGS, 'frames')
SCREEN_WIDTH = getattr(FLAGS, 'screen_width')
SCREEN_HEIGHT = getattr(FLAGS, 'screen_height')
OBV_STEPS = getattr(FLAGS, 'observe_steps')
CKP_STEP = getattr(FLAGS, 'check_point_steps')

MAX_EP = getattr(FLAGS, 'episodes')
EP_STEPS = getattr(FLAGS, 'epochs')

DIM_STATE = [SCREEN_WIDTH, SCREEN_HEIGHT, STATE_FRAMES]
DIM_ACTION = getattr(FLAGS, 'dim_action')

LR = getattr(FLAGS, 'learning_rate')
GAMMA = getattr(FLAGS, 'gamma')

DIR_SUM = getattr(FLAGS, 'dir_sum').format(time_stamp)
DIR_MOD = getattr(FLAGS, 'dir_mod').format(time_stamp)

BUFFER_SIZE = getattr(FLAGS, 'size_buffer')
MINI_BATCH = getattr(FLAGS, 'mini_batch')

EPS_BEGIN = getattr(FLAGS, 'epsilon_begin')
EPS_END = getattr(FLAGS, 'epsilon_end')
EPS_STEPS = getattr(FLAGS, 'epsilon_steps')

DISPLAY = getattr(FLAGS, 'display')


class DqnCatcher():
    def __init__(self, playback_mode, env, render=True, mod=None):
        self._playback_mode = playback_mode

        self._env = env
        self._render = render

        self._sess = tf.Session()
        self._agent = DQNAgent(self._sess, DIM_STATE, DIM_ACTION, LR, net_name='cnn')
        self._sess.run(tf.global_variables_initializer())

        self._saver = tf.train.Saver()
        self._replay_buffer = ReplayBuffer(BUFFER_SIZE)
        self._explorer = Explorer(EPS_BEGIN, EPS_END, EPS_STEPS, playback_mode)
        self.summary = Summary(self._sess, DIR_SUM)

        self.summary.add_variable(tf.Variable(0.), 'reward')
        self.summary.add_variable(tf.Variable(0.), 'loss')
        self.summary.add_variable(tf.Variable(0.), 'maxq')
        self.summary.build()
        self.summary.write_variables(FLAGS)

        self._steps = 0

        if mod and os.path.exists(FLAGS.dir_mod.format(mod)):
            checkpoint = tf.train.get_checkpoint_state(FLAGS.dir_mod.format(mod))
            self._saver.restore(self._sess, save_path=checkpoint.model_checkpoint_path)
            print("Loaded checkpoints {0}".format(checkpoint.model_checkpoint_path))

    def start(self):
        for ep in range(MAX_EP):

            sum_reward = 0
            last_state = []
            last_img = self._env.reset()
            last_img = (pre_process_image_catcher(last_img))

            for _ in range(STATE_FRAMES):
                last_state.append(last_img)
            last_state = np.dstack(last_state)

            for step in range(EP_STEPS):
                if self._render:
                    env.render()

                q_value = self._agent.predict([last_state])[0]
                last_max_qvalue = np.max(q_value)

                act_1_hot = self._explorer.get_action(q_value)
                act_index = np.argmax(act_1_hot)

                observation, reward, done, info = env.step(act_index)

                state = pre_process_image_catcher(observation)
                state = np.reshape(state, (SCREEN_WIDTH, SCREEN_HEIGHT, 1))
                state = np.append(state, last_state[:, :, :3], axis=2)

                self._replay_buffer.add(last_state, act_1_hot, reward, state, done)

                loss = None
                if not self._playback_mode and len(self._replay_buffer) > OBV_STEPS:
                    loss = self._train()

                last_state = state
                sum_reward += reward
                self._steps += 1

                if done or step == EP_STEPS - 1:
                    print('| Step: %i' % self._steps,
                          '| Episode: %i' % ep,
                          '| Epoch: %i' % step,
                          '| qvalue: %.5f' % last_max_qvalue,
                          '| Sum_Reward: %i' % sum_reward)
                    if loss != None:
                        self.summary.run(feed_dict={
                            'loss': loss,
                            'reward': sum_reward,
                            'maxq': last_max_qvalue})
                    break

    def _train(self):
        batch_state, batch_action, batch_reward, batch_state_next, batch_done = \
            self._replay_buffer.sample_batch(MINI_BATCH)

        q_value = self._agent.predict(batch_state_next)

        batch_y = []
        for r, q, d in zip(batch_reward, q_value, batch_done):
            if d:
                batch_y.append(r)
            else:
                batch_y.append(r + GAMMA * np.max(q))

        act_q_value, loss = self._agent.train(batch_state, batch_action, batch_y)

        if not self._steps % CKP_STEP:
            self._saver.save(self._sess, DIR_MOD + '/net', global_step=self._steps)
            print('Mod saved!')

        return loss


if __name__ == '__main__':
    # print(env.unwrapped.get_action_meanings())
    # os.environ['CUDA_VISIBLE_DEVICES'] = ''
    env = gym.make('Catcher-v0')
    dqn = DqnCatcher(playback_mode=False, render=False, env=env, mod='')
    dqn.start()
