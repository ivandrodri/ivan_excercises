#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

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


#task = "Reacher-v5"#"Humanoid-v4"#"Ant-v3"
task = "CartPole-v0"
seed = 0
hidden_sizes =[256, 256]
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.99
tau = 0.005
alpha = 0.2



n_step = 1 #### Not very clear what it is --> see sppining bla bla
training_num = 1
test_num = 1
log_dir = "log"
render_flag = 1.0
device = "cpu"
resume_path = "log/Reacher-v4/sac/0/230725-211942/policy.pth"
logger_id = "tensorboard"


def ivan_sac_lab():

    import gymnasium as gym
    gym.envs.register(
        id="Reacher-v5",
        entry_point="gymnasium.envs.mujoco.reacher_v4:ReacherEnv",
        max_episode_steps=60,
        reward_threshold=-3.75,
    )

    env, train_envs, test_envs = make_mujoco_env(
        task, seed, training_num, test_num, obs_norm=False, render="rgb_array",
    )

    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n
#    max_action = env.action_space.high[0]
    print("Observations shape:", state_shape)
    print("Actions shape:", action_shape)
    #print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))
    # seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # model
    net_a = Net(state_shape, hidden_sizes=hidden_sizes, device=device)
    actor = ActorProb(
        net_a,
        action_shape,
        device=device,
        unbounded=True,
        conditioned_sigma=True,
    ).to(device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    net_c1 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    net_c2 = Net(
        state_shape,
        action_shape,
        hidden_sizes=hidden_sizes,
        concat=True,
        device=device,
    )
    critic1 = Critic(net_c1, device=device).to(device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=critic_lr)
    critic2 = Critic(net_c2, device=device).to(device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=critic_lr)

    policy = SACPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        tau=tau,
        gamma=gamma,
        alpha=alpha,
        estimation_step=n_step,
        action_space=env.action_space,
    )

    # load a previous policy
    policy.load_state_dict(torch.load(resume_path, map_location=device))
    print("Loaded agent from: ", resume_path)

    test_collector = Collector(policy, test_envs)
    # Let's watch its performance after every epoch!
    policy.eval()
    test_envs.seed(seed)
    test_collector.reset()
    result = test_collector.collect(n_step=5000, render=render_flag )
    print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


#if __name__ == "__main__":
ivan_sac_lab()
