# test the performance of Ly policy (dynamic and static ly policy)
import time
import numpy as np
import gym
from gym.envs.registration import register


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

def Ly_policy(policy_set, env):
    state = env.s
    max_value = -np.inf
    max_index = 0
    pl_vu = []
    pl_vd = []
    t = 0
    for j, i in enumerate(policy_set):
        sinr_dl, sinr_ul = env.get_sinr(list(i), state[0], state[1], state[2], state[3])
        v_d = []
        for i in sinr_dl:
            if i == -np.inf:
                v_d.append(0)
            else:
                v_d.append(env.v[env.get_mode(i)])
        v_u = []
        for i in sinr_ul:
            if i == -np.inf:
                v_u.append(0)
            else:
                v_u.append(env.v[env.get_mode(i)])
        value = np.sum(state[4] * v_u) + np.sum(state[5] * v_d)
        if value > max_value:
            max_value = value
            max_index = j
            pl_vu = v_u
            pl_vd = v_d
    sinr_dl_real, sinr_ul_real = env.get_sinr(policy_set[max_index], state[0], state[1], state[2], state[3])
    v_d = []
    for i in sinr_dl_real:
        if i == -np.inf:
            v_d.append(0)
        else:
            v_d.append(env.v[env.get_mode(i)])
    v_u = []
    for i in sinr_ul_real:
        if i == -np.inf:
            v_u.append(0)
        else:
            v_u.append(env.v[env.get_mode(i)])
    return policy_set[max_index], v_u, v_d


register_multibs_env()
ENV_NAME = 'multi-bs-fd-v0'
env = gym.make(ENV_NAME)
env.reset()
duration = 5000
ly_results = []
static_ly_results = []
'''
policy_set = []
for i in itertools.product(env.action_space, repeat=env.bs_num):
    policy_set.append(i)
'''
static_policy_set = []
for i in env.action_space:
    static_policy_set.append([i]*env.bs_num)
'''
episode_time = time.time()
for i in range(duration):
    policy, _, _ = Ly_policy(policy_set, env)
    _, reward, _, _ = env.step(policy)
    ly_results.append(-reward)
    if i % 100 == 0:
        print(i)
print('Ly took %.5f' % (time.time() - episode_time))
'''
for i in range(50):
    env.reset()
    ans = []
    episode_time = time.time()
    for i in range(duration):
        policy, _, _ = Ly_policy(static_policy_set, env)
        _, reward, _, _ = env.step(policy)
        ans.append(-reward)
        if i % 1000 == 0:
            print(i)
    print('static Ly took %.5f' % (time.time() - episode_time))
    print('static Ly reward %.5f' % np.mean(ans))
    static_ly_results.append(np.mean(ans))
print('static Ly average reward %.5f' % np.mean(static_ly_results))

## CEM? may be works well?