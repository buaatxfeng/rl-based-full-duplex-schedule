# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 13:21:43 2018

@author: TXF
"""

'''
Implement a new gym-like single-agent environment/scenarios by implementing the _reset, _step, _seed function 
'''

import numpy as np
from gym import Env, spaces

class SingleAgentMultiBs(Env):
    # indicate action space
    FREE = 0
    UP_LINK = 1
    DOWN_LINK = 2
    FULL_LINK = 3
    def __init__(self, bandwidth, slot, Pb, Pu, packet_size, para_table, self_reduction,
                 carrier_frequency, antenna_bs, antenna_user, col_bs, row_bs, ISD, lamba_u, lamba_d, penetration=20, full_duplex=False,
                 pos_bs=None, pos_user=None):
        self.bandwidth = bandwidth  # 带宽 HZ
        self.slot = slot  # slot长度ms
        self.Pb = Pb  # 基站发射功率
        self.Pu = Pu  # 基站接收功率
        self.packet_size = packet_size  # 包大小（bits）
        self.para_table = para_table  # mcs table
        self.carrier_frequency = carrier_frequency  # 载频，GHZ
        self.full_duplex = full_duplex  # 系统是否支持全双工
        self.self_reduction = self_reduction  # 基站自损耗系数
        self.antenna_bs = antenna_bs  # 基站天线数
        self.antenna_user = antenna_user  # 用户端天线数
        self.col_bs = col_bs  # 小区列数
        self.row_bs = row_bs  # 小区行数
        self.bs_num = self.col_bs * self.row_bs  # 基站数目
        self.ISD = ISD  # 正方形小区长宽（m）
        self.lamba_u = lamba_u
        self.lamba_d = lamba_d
        # 每个level的传输速率（packets/s）
        self.v = self.para_table[1, :] * self.bandwidth * self.slot / self.packet_size
        # 决定基站和用户位置（1.随机撒点 2.给出位置数据）
        if pos_bs is None and pos_user is None:
            self.pos_bs, self.pos_user = self.determin_position()
        else:
            self.pos_bs = pos_bs
            self.pos_user = pos_user
        self.penetration = penetration  # 穿墙损耗
        self.nfigure_bs = 7  # noise figure at BS, in dB
        self.nfigure_user = 13  # noise figure at UE, in dB
        # 基站和用户端的噪声功率（线性）
        self.p_noise_bs = np.power(10, (-174 + 10 * np.log10(self.bandwidth) + self.nfigure_bs) / 10)
        self.p_noise_user = np.power(10, (-174 + 10 * np.log10(self.bandwidth) + self.nfigure_user) / 10)
        #        self.CH_BB, self.CH_BU_DL, self.CH_BU_UL, self.CH_UU = self.generate_channels()
        if not full_duplex:
            self.action_space = np.arange(3)
            self.na = pow(3, self.bs_num)
        else:
            self.action_space = np.arange(4)
            self.na = pow(4, self.bs_num)

        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(4*self.bs_num*self.bs_num + 2*self.bs_num,))
        self.ns_dim = 4*self.bs_num*self.bs_num + 2*self.bs_num


    def _reset(self):
        """Reset the environment.

        Returns
        -------
          A tuple representing the current state with meanings
          (uplink_hd_rate, downlink_hd_rate, uplink_fd_rate, downlink_fd_rate, queue_u=0, queue_d=0, arrive_u, arrive_d).
        """
        '''
        h_all = np.linalg.norm(np.random.normal(loc=0.0, scale=1.0, size=(3, 2)), axis=1)
        h_all = h_all * h_all * 0.5
        '''

        power_dl_from_bs, power_dl_from_user, power_ul_from_user, power_ul_from_bs = self.generate_power()
        uplink_queue = np.zeros(self.bs_num)
        downlink_queue = np.zeros(self.bs_num)
        s = tuple([power_dl_from_bs, power_dl_from_user, power_ul_from_user, power_ul_from_bs, uplink_queue, downlink_queue])
        self.s = s
        return s


    def _step(self, actions):
        """Execute the specified action.

        Parameters
        ----------
        action: int
          A number in range [0, 3] or [0, 2]. Represents the action.

        Returns
        -------
        (state, reward, is_terminal, debug_info)
          State is the tuple in the same format as the reset
          method. Reward is a floating point number. is_terminal is a
          boolean representing if the new state is a terminal
          state. debug_info is a dictionary. You can fill debug_info
          with any additional information you deem useful.
        """
        state = self.s
        if type(actions) is not list and type(actions) is not tuple and type(actions) is not np.ndarray:
            action = []
            if self.full_duplex:
                a = 4
            else:
                a = 3
            while actions != 0:
                tem = actions % a
                actions = actions // a
                action.append(tem)
            while len(action) != 6:
                action.append(0)
        else:
            action = actions
        sinr_dl, sinr_ul = self.get_sinr(action, state[0], state[1], state[2], state[3])
        # generate new power
        power_dl_from_bs, power_dl_from_user, power_ul_from_user, power_ul_from_bs = self.generate_power()
        # generate new samples
        samples_u, samples_d = self.generate_sample(self.lamba_u, self.lamba_d)

        q_u = state[4]
        q_d = state[5]

        current_qu = np.zeros(self.bs_num)
        current_qd = np.zeros(self.bs_num)
        # update queue length for each cell
        for i in range(self.bs_num):
            if action[i] == SingleAgentMultiBs.FREE:
                current_qu[i] = max(0, q_u[i]) + samples_u[i]
                current_qd[i] = max(0, q_d[i]) + samples_u[i]
            elif action[i] == SingleAgentMultiBs.UP_LINK:
                current_qu[i] = max(0, q_u[i] + samples_u[i] - self.v[self.get_mode(sinr_ul[i])])
                current_qd[i] = max(0, q_d[i]) + samples_d[i]
            elif action[i] == SingleAgentMultiBs.DOWN_LINK:
                current_qu[i] = max(0, q_u[i] + samples_u[i])
                current_qd[i] = max(0, q_d[i] + samples_d[i] - self.v[self.get_mode(sinr_dl[i])])
            else:
                current_qu[i] = max(0, q_u[i] + samples_u[i] - self.v[self.get_mode(sinr_ul[i])])
                current_qd[i] = max(0, q_d[i] + samples_d[i] - self.v[self.get_mode(sinr_dl[i])])

        next_state = tuple([power_dl_from_bs, power_dl_from_user, power_ul_from_user, power_ul_from_bs, current_qu, current_qd])
        # define reward as current queue length
        # reward =  np.sum([next_state[4][i] + next_state[5][i] for i in range(self.bs_num)])
        reward = np.sum([next_state[4][i]/self.lamba_u[i] + next_state[5][i]/self.lamba_d[i] for i in range(self.bs_num)])
        is_terminal = False
        self.s = next_state
        return next_state, -reward, is_terminal, None

    def _seed(self, seed=None):
        np.random.seed(seed)
        return [seed]

    def determin_position(self):
        """
        :return: 随机撒点，确定每个小区基站和上下行用户的位置
        pos_user[:,0] for down link user
        pos_user[:,1] for up link user
        """
        pos_bs = np.zeros(self.bs_num, dtype=np.complex)
        pos_user = np.zeros((self.bs_num, 2), dtype=np.complex)
        # 0 for up link user, 1 for down link user
        for i in range(self.bs_num):
            row = int(i / self.col_bs)
            col = i % self.col_bs
            pos_bs[i] = complex(col * self.ISD, row * self.ISD)
            while True:
                pos_user[i, 0] = complex((np.random.rand() - 0.5) * self.ISD + pos_bs[i].real,
                                         (np.random.rand() - 0.5) * self.ISD + pos_bs[i].imag)
                if np.abs(pos_user[i, 0] - pos_bs[i]) > 3:
                    break
            while True:
                pos_user[i, 1] = complex((np.random.rand() - 0.5) * self.ISD + pos_bs[i].real,
                                         (np.random.rand() - 0.5) * self.ISD + pos_bs[i].imag)
                if np.abs(pos_user[i, 1] - pos_bs[i]) > 3 and np.abs(pos_user[i, 1] - pos_user[i, 0]) > 3:
                    break
        return pos_bs, pos_user

    def path_loss(self, distance, los_or_nlos):
        """
        :param distance:
        :param los_or_nlos:
        :return: 根据距离计算路径损耗(db)
        """
        if los_or_nlos == 'LOS':
            PL = 32.4 + 17.3 * np.log10(distance) + 20 * np.log10(self.carrier_frequency)
        else:
            PL1 = 32.4 + 17.3 * np.log10(distance) + 20 * np.log10(self.carrier_frequency)
            PL2 = 17.3 + 38.3 * np.log10(distance) + 24.9 * np.log10(self.carrier_frequency)
            PL = max(PL1, PL2)
        return PL

    def power_computing(self, p, los_or_nlos, pos_from, pos_to, ans_from, ans_to):
        """
        根据发射功率，是否los，发射端，接收端的位置和天线数计算接收功率
        """
        distance = np.abs(pos_from - pos_to)
        PL = self.path_loss(distance, los_or_nlos)
        if not los_or_nlos:
            PL += self.penetration
        h = np.linalg.norm(np.random.normal(loc=0.0, scale=1.0, size=(1, 2)), axis=1)
        h = h * h * 0.5
        power = p - PL + 10 * np.log10(ans_from * ans_to) + 10 * np.log10(h)
        return np.power(10, 0.1 * power)

    def generate_power(self):
        """
        :return: 给定四种信道，生成对应的四种功率，都是6*6的数组
        """
        # 基站到下行用户的功率（信号或者干扰）
        power_dl_from_bs = np.zeros((self.bs_num, self.bs_num))
        for i in range(self.bs_num):  # Channels to UE_i_0
            for j in range(self.bs_num):  # Channels from bs_j
                if i == j:
                    power_dl_from_bs[i, j] = self.power_computing(self.Pb,
                                                                  True, self.pos_bs[j], self.pos_user[i, 0],
                                                                  self.antenna_bs, self.antenna_user)
                else:
                    power_dl_from_bs[i, j] = self.power_computing(self.Pb,
                                                                  False, self.pos_bs[j], self.pos_user[i, 0],
                                                                  self.antenna_bs, self.antenna_user)

        # 上行用户到下行用户的功率（干扰）
        power_dl_from_user = np.zeros((self.bs_num, self.bs_num))
        for i in range(self.bs_num):  # Channels to UE_i_0
            for j in range(self.bs_num):  # Channels from UE_j_1
                if i == j:
                    power_dl_from_user[i, j] = self.power_computing(self.Pu,
                                                                    True, self.pos_user[j, 1], self.pos_user[i, 0],
                                                                    self.antenna_user, self.antenna_user)
                else:
                    power_dl_from_user[i, j] = self.power_computing(self.Pb,
                                                                    False, self.pos_user[j, 1], self.pos_user[i, 0],
                                                                    self.antenna_user, self.antenna_user)

        # 上行用户到基站的功率（信号和干扰）
        power_ul_from_user = np.zeros((self.bs_num, self.bs_num))
        for i in range(self.bs_num):  # Channels to bS_i
            for j in range(self.bs_num):  # Channels from UE_j_1
                if i == j:
                    power_ul_from_user[i, j] = self.power_computing(self.Pu,
                                                                    True, self.pos_user[j, 1], self.pos_bs[i],
                                                                    self.antenna_user, self.antenna_bs)
                else:
                    power_ul_from_user[i, j] = self.power_computing(self.Pu,
                                                                    False, self.pos_user[j, 1], self.pos_bs[i],
                                                                    self.antenna_user, self.antenna_bs)

        # 基站到基站的功率 （干扰）
        power_ul_from_bs = np.zeros((self.bs_num, self.bs_num))
        for i in range(self.bs_num):  # Channels to UL BS_i
            for j in range(self.bs_num):  # Channels from DL BS_j
                if i == j:
                    power_ul_from_bs[i, j] = 0
                else:
                    power_ul_from_bs[i, j] = self.power_computing(self.Pu,
                                                                  False, self.pos_bs[j], self.pos_bs[i],
                                                                  self.antenna_bs, self.antenna_bs)

        return power_dl_from_bs, power_dl_from_user, power_ul_from_user, power_ul_from_bs

    def get_sinr(self, policy, power_dl_from_bs, power_dl_from_user, power_ul_from_user, power_ul_from_bs):
        """
        :param policy: 每个基站当前的policy
        :param power_dl_from_bs:
        :param power_dl_from_user:
        :param power_ul_from_user:
        :param power_ul_from_bs:
        :return: 当前的信噪比/信干噪比, 按照调制参数表，当前policy下每个基站上下行的速率
        """
        sinr_dl = np.zeros(self.bs_num)
        sinr_ul = np.zeros(self.bs_num)
        if not self.full_duplex:
            for i in range(self.bs_num):
                if policy[i] == 2:
                    PS = power_dl_from_bs[i, i]
                    PI_BS = 0
                    PI_USER = 0
                    for j in range(self.bs_num):
                        if j != i and policy[j] == 2:
                            PI_BS += power_dl_from_bs[i, j]
                        elif j != i and policy[j] == 1:
                            PI_USER += power_dl_from_user[i, j]
                    sinr_dl[i] = 10 * np.log10(PS / (PI_BS + PI_USER + self.p_noise_user))
                    sinr_ul[i] = - np.inf
                elif policy[i] == 1:
                    PS = power_ul_from_user[i, i]
                    PI_BS = 0
                    PI_USER = 0
                    for j in range(self.bs_num):
                        if j != i and policy[j] == 2:
                            PI_BS += power_ul_from_bs[i, j]
                        elif j != i and policy[j] == 1:
                            PI_USER += power_ul_from_user[i, j]
                    sinr_ul[i] = 10 * np.log10(PS / (PI_BS + PI_USER + self.p_noise_bs))
                    sinr_dl[i] = - np.inf
                else:
                    sinr_ul[i] = - np.inf
                    sinr_dl[i] = - np.inf
        else:
            for i in range(self.bs_num):
                if policy[i] == 2:
                    PS = power_dl_from_bs[i, i]
                    PI_BS = 0
                    PI_USER = 0
                    for j in range(self.bs_num):
                        if j != i and policy[j] == 2:
                            PI_BS += power_dl_from_bs[i, j]
                        elif j != i and policy[j] == 1:
                            PI_USER += power_dl_from_user[i, j]
                        elif j != i and policy[j] == 3:
                            PI_BS += power_dl_from_bs[i, j]
                            PI_USER += power_dl_from_user[i, j]
                    sinr_dl[i] = 10 * np.log10(PS / (PI_BS + PI_USER + self.p_noise_user))
                    sinr_ul[i] = - np.inf
                elif policy[i] == 1:
                    PS = power_ul_from_user[i, i]
                    PI_BS = 0
                    PI_USER = 0
                    for j in range(self.bs_num):
                        if j != i and policy[j] == 2:
                            PI_BS += power_ul_from_bs[i, j]
                        elif j != i and policy[j] == 1:
                            PI_USER += power_ul_from_user[i, j]
                        elif j != i and policy[j] == 3:
                            PI_BS += power_ul_from_bs[i, j]
                            PI_USER += power_ul_from_user[i, j]
                    sinr_ul[i] = 10 * np.log10(PS / (PI_BS + PI_USER + self.p_noise_bs))
                    sinr_dl[i] = - np.inf
                elif policy[i] == 3:
                    PS_d = power_dl_from_bs[i, i]
                    PS_u = power_ul_from_user[i, i]
                    PI_SELF = np.power(10, (self.Pb - self.self_reduction) / 10)
                    PI_BS_U = PI_SELF
                    PI_USER_U = 0
                    PI_BS_D = 0
                    PI_USER_D = power_dl_from_user[i, i]
                    for j in range(self.bs_num):
                        if j != i and policy[j] == 2:
                            PI_BS_D += power_dl_from_bs[i, j]
                            PI_BS_U += power_ul_from_bs[i, j]
                        elif j != i and policy[j] == 1:
                            PI_USER_D += power_dl_from_user[i, j]
                            PI_USER_U += power_ul_from_user[i, j]
                        elif j != i and policy[j] == 3:
                            PI_BS_D += power_dl_from_bs[i, j]
                            PI_BS_U += power_ul_from_bs[i, j]
                            PI_USER_D += power_dl_from_user[i, j]
                            PI_USER_U += power_ul_from_user[i, j]
                    sinr_dl[i] = 10 * np.log10(PS_d / (PI_BS_D + PI_USER_D + self.p_noise_user))
                    sinr_ul[i] = 10 * np.log10(PS_u / (PI_BS_U + PI_USER_U + self.p_noise_bs))
                else:
                    sinr_ul[i] = - np.inf
                    sinr_dl[i] = - np.inf
        return sinr_dl, sinr_ul

    def get_mode(self, snr):
        # 根据信道质量，判断进行哪种方式的调制
        k = self.para_table.shape[1]
        if snr < self.para_table[0, 0]:
            return 0
        if snr > self.para_table[0, k - 1]:
            return k - 1
        for i in range(1, k):
            if snr <= self.para_table[0, i]:
                if snr > self.para_table[0, i - 1]:
                    return i

    def generate_sample(self, lamba_u, lamba_d):
        """
        :param lamba_u:
        :param lamba_d:
        :return: # 给定到达率生成样本
        """
        lamba_u_p = np.array([i * self.slot for i in lamba_u])
        lamba_d_p = np.array([i * self.slot for i in lamba_d])
        # 采样上下行packet流
        samples_u = np.random.poisson(lamba_u_p)
        samples_d = np.random.poisson(lamba_d_p)
        return samples_u, samples_d

