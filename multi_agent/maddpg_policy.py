#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 14:55
# @Author  : Xiaofeng Tan
# @Site    : 
# @File    : maddpg_policy.py
# @Software: PyCharm
"""
implement an MADDPG algorithm for multi agent env
Multi agent deep deterministic policy gradient , Reinforcement Learning.
"""

import argparse
import numpy as np
import tensorflow as tf
import gym
import os
import time
import pickle
from gym.envs.registration import register
import maddpg.common.tf_util as U
from maddpg.trainer.maddpg import MADDPGAgentTrainer
import tensorflow.contrib.layers as layers


## 定义命令行参数
def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning for multiagent environments")
    # Environment
    parser.add_argument("--max-episode-len", type=int, default=5000, help="maximum episode length")
    parser.add_argument("--num-episodes", type=int, default=500, help="number of episodes")
    parser.add_argument("--num-test-episodes", type=int, default=100, help="number of test episodes")
    parser.add_argument("--num-agents", type=int, default=6, help="number of agents")
    parser.add_argument("--policy", type=str, default="maddpg", help="policy for each agent")
    # Core training parameters
    parser.add_argument("--lr-p", type=float, default=1e-4, help="learning rate for actor optimizer")
    parser.add_argument("--lr-q", type=float, default=1e-3, help="learning rate for critic optimizer")
    parser.add_argument("--gamma", type=float, default=0.8, help="discount factor")
    parser.add_argument("--batch-size", type=int, default=128, help="number of episodes to optimize at the same time")
    parser.add_argument("--num-units", type=int, default=200, help="number of units in the mlp")
    # Checkpointing
    parser.add_argument("--exp-name", type=str, default='multi-bs-fd-v0', help="name of the experiment")
    parser.add_argument("--save-dir", type=str, default="./policy4/model.ckpt",
                        help="directory in which training state and model should be saved")
    parser.add_argument("--save-rate", type=int, default=1,
                        help="save model once every time this many episodes are completed")
    parser.add_argument("--load-dir", type=str, default="./policy4/model.ckpt",
                        help="directory in which training state and model are loaded")
    # Train or Test
    parser.add_argument("--is-train", type=bool, default=True, help="train or test")
    parser.add_argument("--reuse", type=bool, default=False, help="reuse or not")
    parser.add_argument("--plots-dir", type=str, default="./learning_curves/",
                        help="directory where plot data is saved")
    return parser.parse_args()


def mlp_model(input, num_outputs, scope, reuse=False, is_train=True, is_p=True, num_units=64, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.batch_norm(out, is_training=is_train)
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.batch_norm(out, is_training=is_train)
        out = layers.fully_connected(out, num_outputs=num_units // 2, activation_fn=tf.nn.relu)
        out = layers.batch_norm(out, is_training=is_train)
        if is_p:
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=tf.nn.softmax)
        else:
            out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)
        return out


## 注册环境
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
        entry_point='multiagent_env:MultiAgentMultiBS',
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
        entry_point='multiagent_env:MultiAgentMultiBS',
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


def make_env(name):
    register_multibs_env()
    env = gym.make(name)
    return env


def get_trainers(env, obs_shape_n, arglist):
    trainers = []
    model = mlp_model
    trainer = MADDPGAgentTrainer
    for i in range(arglist.num_agents):
        trainers.append(trainer("agent_%d" % i, model, obs_shape_n, env.action_space, i, arglist,
                                is_train=is_train, local_q_func=(arglist.policy == 'ddpg')))
    return trainers


def train(arglist):
    with U.make_session(2):
        # Create environment
        env = make_env(arglist.exp_name)
        # Create agent trainers
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        trainers = get_trainers(env, obs_shape_n, arglist)
        print('Using policy {}'.format(arglist.policy))

        # Initialize
        U.initialize()

        # Load previous results, if necessary
        # reuse or not
        if arglist.load_dir == "":
            arglist.load_dir = arglist.save_dir
        if arglist.reuse:
            print('Loading previous state...')
            U.load_state(arglist.load_dir)
            print('Loading successfully!')

        # generate log path, if neccessary
        if not os.path.exists(arglist.plots_dir + arglist.exp_name):
            os.makedirs(arglist.plots_dir + arglist.exp_name)

        if arglist.is_train:
            episode_rewards = [0.0]  # sum of rewards for all agents
            agent_rewards = [[0.0] for _ in range(env.n)]  # individual agent reward
            final_ep_rewards = []  # sum of rewards for training curve
            final_ep_ag_rewards = []  # agent rewards for training curve
            agent_info = [[[]]]  # placeholder for benchmarking info
            saver = tf.train.Saver()
            obs_n = env.reset()
            episode_step = 0
            train_step = 0
            t_start = time.time()

            print('Starting iterations...')
            while True:
                # get action
                action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                # choose action in environment
                # environment step
                new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                episode_step += 1
                done = all(done_n)
                terminal = (episode_step >= arglist.max_episode_len)
                # collect experience
                for i, agent in enumerate(trainers):
                    agent.experience(obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                obs_n = new_obs_n

                for i, rew in enumerate(rew_n):
                    episode_rewards[-1] += rew
                    agent_rewards[i][-1] += rew

                if done or terminal:
                    obs_n = env.reset()
                    episode_step = 0
                    episode_rewards.append(0)
                    for a in agent_rewards:
                        a.append(0)
                    agent_info.append([[]])

                # increment global step counter
                train_step += 1

                # update all trainers, if not in display or benchmark mode
                loss = None
                for agent in trainers:
                    agent.preupdate()
                for agent in trainers:
                    loss = agent.update(trainers, train_step)

                # save model, display training output
                if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                    U.save_state(arglist.save_dir, len(episode_rewards) - 1, saver=saver)
                    # print statement depends on whether or not there are adversaries
                    print("steps: {}, episodes: {}, mean episode reward: {}, time: {}".format(
                        train_step, len(episode_rewards) - 1,
                                    np.mean(episode_rewards[-arglist.save_rate - 1:-1]) / arglist.max_episode_len,
                        round(time.time() - t_start, 3)))
                    print('save model!')
                    t_start = time.time()
                    # Keep track of final episode reward
                    final_ep_rewards.append(np.mean(episode_rewards[-arglist.save_rate:]))
                    for rew in agent_rewards:
                        final_ep_ag_rewards.append(np.mean(rew[-arglist.save_rate:]))

                # saves final episode reward for plotting training curve later
                if len(episode_rewards) > arglist.num_episodes:
                    rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards.pkl'
                    with open(rew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_rewards, fp)
                    agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards.pkl'
                    with open(agrew_file_name, 'wb') as fp:
                        pickle.dump(final_ep_ag_rewards, fp)
                    print('...Finished total of {} episodes.'.format(len(episode_rewards)))
                    break

        else:
            episode_rewards = []
            obs_n = env.reset()
            print('Starting test...')
            for i in range(arglist.num_test_episodes):
                episode_step = 0
                tem_rew = 0
                t_start = time.time()
                while True:
                    # get action
                    action_n = [agent.action(obs) for agent, obs in zip(trainers, obs_n)]
                    # choose action in environment
                    # environment step
                    # do greedy chooice in test mode?
                    '''
                    greedy_action_n = [[0]*len(i) for i in action_n]
                    for i,act in enumerate(action_n):
                        greedy_action_n[i][np.argmax(act)] = 1
                    '''
                    new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                    episode_step += 1
                    done = all(done_n)
                    terminal = (episode_step >= arglist.max_episode_len)
                    tem_rew += sum(rew_n)

                    if done or terminal:
                        obs_n = env.reset()
                        episode_rewards.append(tem_rew / episode_step)
                        print("episodes: {}, mean episode reward: {}, time: {}".format(
                            i, episode_rewards[-1], round(time.time() - t_start, 3)))
                        break


if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)