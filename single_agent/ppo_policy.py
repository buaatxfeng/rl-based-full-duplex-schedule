#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2019/6/8 14:55
# @Author  : Xiaofeng Tan
# @Site    : 
# @File    : ppo_policy.py
# @Software: PyCharm

"""
implement an PPO algorithm for single agent env with a 4*N (or 3*N for HD) node output layer
Proximal policy optimization, Reinforcement Learning.
"""
from collections import namedtuple
import os
import time
import numpy as np

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from tensorboardX import SummaryWriter

from gym.envs.registration import register


# register env
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
                'col_bs': 3,
                'row_bs': 2,
                'ISD': 20,
                'lamba_u': lamba_u,
                'lamba_d': lamba_d,
                'penetration': 20,
                'full_duplex': False,
                })

    register(
        id='multi-bs-fd-v0',
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
                'full_duplex': True,
                })


# Parameters
gamma = 1
render = False
# seed = 1
log_interval = 10

register_multibs_env()
ENV_NAME = 'multi-bs-fd-v0'
env = gym.make(ENV_NAME)

num_state = env.ns_dim
num_action = env.na
na_per_bs = 4 if env.full_duplex else 3
# torch.manual_seed(seed)
# env.seed(seed)
Transition = namedtuple('Transition', ['state', 'action', 'a_log_prob', 'reward', 'next_state'])


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.action_head = [nn.Linear(100, na_per_bs) for i in range(env.bs_num)]
        # self.action_head = nn.Linear(100, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = torch.stack([self.action_head[i](x) for i in range(env.bs_num)], -2)
        action_prob = F.softmax(x, dim=-1)
        # dim(?, env.bs_num, 4)
        return action_prob


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 100)
        self.state_value = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        value = self.state_value(x)
        return value


class PPO():
    clip_param = 0.2
    max_grad_norm = 1
    ppo_update_time = 10
    buffer_capacity = 5000
    batch_size = 256

    def __init__(self):
        super(PPO, self).__init__()
        self.actor_net = Actor()
        self.critic_net = Critic()
        self.buffer = []
        self.counter = 0
        self.training_step = 0
        self.writer = SummaryWriter(r'.\exp')

        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), 1e-5)
        self.critic_net_optimizer = optim.Adam(self.critic_net.parameters(), 3e-5)
        if not os.path.exists(r'.\param'):
            os.makedirs(r'.\param\net_param')
            os.makedirs(r'.\param\img')

    def select_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            action_prob = self.actor_net(state).squeeze()
        c = Categorical(action_prob)
        action = c.sample()
        prob = torch.gather(action_prob, -1, action.unsqueeze(-1))
        return action.numpy(), torch.prod(prob.squeeze(), dim=-1).item()

    def get_value(self, state):
        state = torch.from_numpy(state)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()

    def save_param(self):
        torch.save(self.actor_net.state_dict(), '.\param\\net_param\\actor_net' + str(time.time())[:10] + '.pkl')
        torch.save(self.critic_net.state_dict(), '.\param\\net_param\critic_net' + str(time.time())[:10] + '.pkl')

    def store_transition(self, transition):
        self.buffer.append(transition)
        self.counter += 1

    def update(self, i_ep):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float32)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long)
        reward = [t.reward for t in self.buffer]
        # update: don't need next_state
        # reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        # next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float32).view(-1, 1)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float32)
        print("The agent is updateing....")
        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                '''
                if self.training_step % 100 ==0:
                    print('I_ep {} ，train {} times'.format(i_ep,self.training_step))
                '''
                # with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = torch.gather(self.actor_net(state[index]), 2, action[index, :].unsqueeze(-1))
                action_prob = torch.prod(action_prob.squeeze(), dim=-1)
                ratio = (action_prob / old_action_log_prob.squeeze()[index])
                surr1 = ratio * advantage.squeeze()
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()  # MAX->MIN desent
                self.writer.add_scalar('loss/action_loss', action_loss, global_step=self.training_step)
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                # update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.writer.add_scalar('loss/value_loss', value_loss, global_step=self.training_step)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:]  # clear experience, on policy！


def main():
    agent = PPO()
    for i_epoch in range(1000):
        episode_time = time.time()
        state = env.reset()
        state = np.concatenate([i.flatten() for i in state])
        if render:
            env.render()
        epoch_reward = 0
        for t in range(5000):
            action, action_prob = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.concatenate([i.flatten() for i in next_state])
            trans = Transition(state, action, action_prob, reward, next_state)
            if render:
                env.render()
            agent.store_transition(trans)
            state = next_state
            epoch_reward += reward

        if len(agent.buffer) >= agent.batch_size:
            agent.update(i_epoch)

        print("Episode: %d reward: %f took: %.5f" % \
              (i_epoch, epoch_reward / 5000, time.time() - episode_time))

        agent.writer.add_scalar('liveTime/livestep', t, global_step=i_epoch)
        if i_epoch % 10 == 0:
            agent.save_param()


if __name__ == '__main__':
    main()
    print("end")

