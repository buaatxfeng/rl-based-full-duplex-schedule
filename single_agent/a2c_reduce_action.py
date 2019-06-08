#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 14:50
# @Author  : Xiaofeng Tan
# @Site    : 
# @File    : a2c_reduce_action.py
# @Software: PyCharm

"""
implement an A2c algorithm for single agent env with a 4*N (or 3*N for HD) node output layer
Actor-Critic using TD-error as the Advantage, Reinforcement Learning.
"""

import time
import numpy as np
import tensorflow as tf
import gym
import tensorlayer as tl
from tensorlayer.layers import DenseLayer, InputLayer
from gym.envs.registration import register
import shutil
import os
import itertools
import matplotlib.pyplot as plt



tf.logging.set_verbosity(tf.logging.DEBUG)
tl.logging.set_verbosity(tl.logging.DEBUG)

# np.random.seed(2)
# tf.set_random_seed(2)  # reproducible

# hyper-parameters

def register_multibs_env():

    snr_th = [-6.71, -5.11, -3.51, -0.879, 0.701, 2.529, 4.606,
              6.431, 8.326, 10.3, 12.22, 14.01, 15.81, 17.68, 19.61]
    f_e = [0.1523, 0.2344, 0.3770, 0.6016, 0.8770, 1.1758, 1.4766, 1.9141,
           2.4063, 2.7305, 3.3223, 3.9023, 4.5234, 5.1152, 5.5547]

    para_table = np.array([snr_th, f_e])

    lamba_u = [5, 20, 35, 50, 65, 80]
    lamba_d = [80, 65, 50, 35, 20, 5]

    register(
        id='multi-bs-hd-v0',
        entry_point='singleagent_env:SingleAgentMultiBs',
        kwargs={'bandwidth': 80e6,
                'slot': 1e-3,
                'Pb': 23,
                'Pu': 23,
                'packet_size': 0.5 * 1e6 * 8,
                'para_table': para_table,
                'self_reduction': 110,
                'carrier_frequency': 30,
                'antenna_bs': 128,
                'antenna_user': 1,
                'col_bs': 2,
                'row_bs': 3,
                'ISD': 20,
                'lamba_u': lamba_u,
                'lamba_d': lamba_d,
                'penetration': 20,
                'full_duplex': False,
    })

    register(
        id='multi-bs-fd-v0',
        entry_point='multi_bs_env:multi_bs',
        kwargs={'bandwidth': 80e6,
                'slot': 1e-3,
                'Pb': 23,
                'Pu': 23,
                'packet_size': 0.5 * 1e6 * 8,
                'para_table': para_table,
                'self_reduction': 110,
                'carrier_frequency': 30,
                'antenna_bs': 128,
                'antenna_user': 1,
                'col_bs': 2,
                'row_bs': 3,
                'ISD': 20,
                'lamba_u': lamba_u,
                'lamba_d': lamba_d,
                'penetration': 20,
                'full_duplex': True,
                })


ENV_NAME = 'multi-bs-fd-v0'

test_mode = False
reuse = False

OUTPUT_GRAPH = False
MAX_EPISODE = 1000
MAX_EP_STEPS = 5000  # maximum time step in one episode
RENDER = False  # rendering wastes time
TEST_EPISODE = 5
LAMBDA = 0.95  # reward discount in TD error
LR_A = 0.0001  # learning rate for actor
LR_C = 0.001  # learning rate for critic
register_multibs_env()
env = gym.make(ENV_NAME)

# env = env.unwrapped

N_F = env.ns_dim
N_A = env.na
na_per_bs = 4 if env.full_duplex else 3
bs = env.bs_num
# env.action_space.sample() random sample

print("observation dimension: %d" % N_F)
print("num of actions: %d" % N_A)


class Actor(object):

    def __init__(self, sess, n_features, lr=0.001):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, [None], "act")
        self.td_error = tf.placeholder(tf.float32, [None], "td_error")  # TD_error

        with tf.variable_scope('Actor'):  # Policy network
            w_init = tf.contrib.layers.xavier_initializer()
            n = InputLayer(self.s, name='in')
            n = DenseLayer(n, n_units=100, act=tf.nn.relu6, W_init=w_init, name='hidden')
            n = DenseLayer(n, n_units=100, act=tf.nn.relu6, W_init=w_init, name='hidden2')
            n = DenseLayer(n, n_units=50, act=tf.nn.relu6, W_init=w_init, name='hidden3')
            n = DenseLayer(n, n_units=na_per_bs*bs, name='Pi')

            self.acts_logits = tf.stack([n.outputs[0, i * na_per_bs:(i + 1) * na_per_bs] for i in range(bs)])
            self.acts_prob = tf.nn.softmax(self.acts_logits, axis=1)


        with tf.variable_scope('loss'):
            each_ce = tf.stack(
                [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.a[i], logits=self.acts_logits[i, :]) for i in
                 range(bs)])
            cross_entropy = tf.reduce_prod(each_ce, name='prod_b')
            self.exp_v = tf.reduce_sum(tf.multiply(cross_entropy, self.td_error))

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.exp_v)

        # Morvan Zhou (the same)
        # with tf.variable_scope('exp_v'):
        #     # log_prob = tf.log(self.acts_prob[0, self.a[0]])
        #     # self.exp_v = tf.reduce_mean(log_prob * self.td_error[0])  # advantage (TD_error) guided loss
        #     self.exp_v = tl.rein.log_weight(probs=self.acts_prob[0, self.a[0]], weights=self.td_error)
        #
        # with tf.variable_scope('train'):
        #     self.train_op = tf.train.AdamOptimizer(lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def learn(self, s, a, td):
        _, exp_v = self.sess.run([self.train_op, self.exp_v], {self.s: [s], self.a: a, self.td_error: td[0]})
        return exp_v

    def choose_action(self, s):
        probs = self.sess.run(self.acts_prob, {self.s: [s]})  # get probabilities of all actions
        return [tl.rein.choice_action_by_probs(i.ravel()) for i in probs]

    def choose_action_greedy(self, s):
        probs = self.sess.run(self.acts_prob, {self.s: [s]})  # get probabilities of all actions
        return [np.argmax(i.ravel()) for i in probs]


class Critic(object):

    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess
        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        with tf.variable_scope('Critic'):  # we use Value-function here, not Action-Value-function
            w_init = tf.contrib.layers.xavier_initializer()
            n = InputLayer(self.s, name='in')
            n = DenseLayer(n, n_units=100, act=tf.nn.relu6, W_init=w_init, name='hidden')
            n = DenseLayer(n, n_units=100, act=tf.nn.relu, W_init=w_init, name='hidden2')
            n = DenseLayer(n, n_units=50, act=tf.nn.relu, W_init=w_init, name='hidden3')
            n = DenseLayer(n, n_units=1, act=None, name='V')
            self.v = n.outputs

        with tf.variable_scope('squared_TD_error'):
            # TD_error = r + lambd * V(newS) - V(S)
            self.td_error = self.r + LAMBDA * self.v_ - self.v
            self.loss = tf.square(self.td_error)

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(lr).minimize(self.loss)

    def learn(self, s, r, s_):
        v_ = self.sess.run(self.v, {self.s: [s_]})
        td_error, _ = self.sess.run([self.td_error, self.train_op], {self.s: [s], self.v_: v_, self.r: r})
        return td_error


gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

actor = Actor(sess, n_features=N_F, lr=LR_A)
# we need a good teacher, so the teacher should learn faster than the actor
critic = Critic(sess, n_features=N_F, lr=LR_C)


outdir = "results_6bs"
saver = tf.train.Saver(tf.all_variables())

tl.layers.initialize_global_variables(sess)
# model_path = r'E:\PG_TDD\results_6bs\multi-bs-fd-v0\model_5k.ckpt'
model_path = r'E:\PG_TDD\models\multi-bs-fd-v0\model'
if reuse:
    saver.restore(sess, model_path)
    # saver.restore(sess, tf.train.latest_checkpoint(model_path))
    print('load model successfully!')


if not os.path.exists(model_path):
    os.makedirs(model_path)

if not test_mode:
    for i_episode in range(MAX_EPISODE):
        episode_time = time.time()
        s = env.reset()
        s = np.concatenate([i.flatten() for i in s])
        t = 0  # number of step in this episode
        all_r = 0  # rewards of all steps
        while t < MAX_EP_STEPS:
            a = actor.choose_action(s)

            s_new, r, done, info = env.step(a)
            s_new = np.concatenate([i.flatten() for i in s_new])
            all_r += r

            td_error = critic.learn(s, r, s_new)  # learn Value-function : gradient = grad[r + lambda * V(s_new) - V(s)]
            a = np.array(a)
            actor.learn(s, a, td_error)  # learn Policy         : true_gradient = grad[logPi(s, a) * td_error]

            s = s_new
            t += 1

        running_reward = all_r / MAX_EP_STEPS
        print("Episode: %d reward: %f took: %.5f" % \
              (i_episode, running_reward, time.time() - episode_time))

        # test every 5 episode
        if i_episode % 5 == 0 and i_episode > 0:
            t = 0
            all_r = 0
            episode_time = time.time()
            for i in range(TEST_EPISODE):
                t = 0
                s = env.reset()
                s = np.concatenate([i.flatten() for i in s])
                while t < MAX_EP_STEPS:
                    a = actor.choose_action(s)
                    s_new, r, done, info = env.step(a)
                    s_new = np.concatenate([i.flatten() for i in s_new])
                    all_r += r
                    s = s_new
                    t += 1

            running_reward = all_r / (MAX_EP_STEPS*TEST_EPISODE)
            print("Evaluation reward: %f took: %.5f" % \
                  (running_reward, time.time() - episode_time))
            # Save the network's parameters after every epoch
        saver.save(sess, model_path, global_step=i_episode)

else:
    ans = []
    for i in range(100):
        # 测试100个episode
        episode_time = time.time()
        pg_results = []
        s = env.reset()
        s = np.concatenate([i.flatten() for i in s])
        for i in range(5000):
            a = actor.choose_action(s)
            s_new, r, done, info = env.step(a)
            s_new = np.concatenate([i.flatten() for i in s_new])
            pg_results.append(-r)
            s = s_new
        print('PG took %.5f' % (time.time() - episode_time))
        print('PG reward %.3f' % np.mean(pg_results))
        ans.append(np.mean(pg_results))
    print('PG average reward %.3f' % np.mean(ans))

