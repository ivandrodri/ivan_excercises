#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import minari
import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic


#def ivan_mujo():
#    task = "maze2d-medium-v0"
#    seed = 0
#    training_num=1
#    test_num=1
#    env, train_envs, test_envs = make_mujoco_env(
#        task, seed, training_num, test_num, obs_norm=False
#    )

import gymnasium as gym

def load_buffer_d4rl(expert_data_task: str) -> ReplayBuffer:
    #dataset = d4rl.qlearning_dataset(gym.make(expert_data_task))
    #minari.download_dataset(expert_data_task)
    dataset = minari.load_dataset(expert_data_task)

    #replay_buffer = ReplayBuffer.from_data(
    #    obs=dataset["observations"],
    #    act=dataset["actions"],
    #    rew=dataset["rewards"],
    #    done=dataset["terminals"],
    #    obs_next=dataset["next_observations"],
    #    terminated=dataset["terminals"],
    #    truncated=np.zeros(len(dataset["terminals"]))
    #)
    #return replay_buffer












#Terminal

#['', '/home/ivan/anaconda3/envs/tianshou/lib/python38.zip', '/home/ivan/anaconda3/envs/tianshou/lib/python3.8',
# '/home/ivan/anaconda3/envs/tianshou/lib/python3.8/lib-dynload', '/home/ivan/.local/lib/python3.8/site-packages',
# '/home/ivan/anaconda3/envs/tianshou/lib/python3.8/site-packages']


#Run

#['/home/ivan/.local/lib/python3.8/site-packages/ray/thirdparty_files',
# '/home/ivan/.local/lib/python3.8/site-packages/ray/pickle5_files',
# '/home/ivan/Documents/GIT_PROJECTS/Tianshou/tianshou/examples/mujoco',
# '/home/ivan/Documents/GIT_PROJECTS/Tianshou',
# '/home/ivan/.local/share/JetBrains/Toolbox/apps/PyCharm-P/ch-0/231.8109.197/plugins/python/helpers/pycharm_display',
# '/home/ivan/anaconda3/envs/tianshou/lib/python38.zip', '/home/ivan/anaconda3/envs/tianshou/lib/python3.8',
# '/home/ivan/anaconda3/envs/tianshou/lib/python3.8/lib-dynload', '/home/ivan/.local/lib/python3.8/site-packages',
# '/home/ivan/anaconda3/envs/tianshou/lib/python3.8/site-packages',
# '/home/ivan/.local/share/JetBrains/Toolbox/apps/PyCharm-P/ch-0/231.8109.197/plugins/python/helpers/pycharm_matplotlib_backend']