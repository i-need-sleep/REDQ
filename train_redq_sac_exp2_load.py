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

if __name__ == '__main__':
    agent = joblib.load('./trained/mr.agent')
    print(agent.policy_net.state_dict())
    print(agent.replay_buffer.obs1_buf)