import gym
import numpy as np
import torch
import time
import sys
from redq_modified.algos.redq_sac import REDQSACAgent
from redq_modified.algos.core import mbpo_epoches, test_agent
from redq_modified.utils.run_utils import setup_logger_kwargs
from redq_modified.utils.bias_utils import log_bias_evaluation
from redq_modified.utils.logx import EpochLogger

import joblib
import matplotlib.pyplot as plt
import seaborn as sns 

if __name__ == '__main__':
    agent = joblib.load('./trained/mr.agent')

    # 2.1: Get load s,a
    s = torch.tensor(agent.replay_buffer.obs1_buf)
    a = torch.tensor(agent.replay_buffer.acts_buf)

    # Obtain Q(s,a)
    q_prediction_list = []
    for q_i in range(agent.num_Q):
        q_prediction = agent.q_net_list[q_i](torch.cat([s, a], 1))
        q_prediction_list.append(q_prediction)
    q_prediction_cat = torch.cat(q_prediction_list, dim=1)
    q_prediction_min = torch.min(q_prediction_cat, dim=1, keepdim=True).values

    # Find the states with the highest/lowest Q
    high_q_idx = torch.argmax(q_prediction_min).item()
    high_q_s = s[high_q_idx, :]
    high_q = q_prediction_min[high_q_idx, 0]

    low_q_idx = torch.argmin(q_prediction_min).item()
    low_q_s = s[low_q_idx, :]
    low_q = q_prediction_min[low_q_idx, 0]

    print(f'max Q: {high_q} at state {high_q_s}')
    print(f'min Q: {low_q} at state {low_q_s}')

    sns.histplot(q_prediction_min.detach().numpy())
    plt.xlabel('Q value')
    plt.savefig('./figures/2-1.png')

    # 2.2
    _, _, _, log_prob_a_tilda, _, _, = agent.policy_net.forward(s, deterministic=False, return_log_prob=True)
    entropy = - log_prob_a_tilda
    plt.clf()
    sns.histplot(entropy.detach().numpy())
    plt.xlabel('Entropy')
    plt.savefig('./figures/2-2.png')
