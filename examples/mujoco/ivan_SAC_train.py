#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint

import gymnasium
import numpy as np
import torch
from mujoco_env import make_mujoco_env
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, ReplayBuffer, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import SACPolicy
from tianshou.trainer import offpolicy_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import Net
from tianshou.utils.net.continuous import ActorProb, Critic
from examples.custom_envs.custom_grid_env_minari import CustomGridEnvGymnasium, custom_grid_policy_minari

task = "ivan_1d_grid-gymnasium-v1"#"Ant-v3"#"Humanoid-v4"#"Ant-v3"
seed = 0
buffer_size = 50
hidden_sizes =[256, 256]
actor_lr = 1e-3
critic_lr = 1e-3
gamma = 0.99
tau = 0.005
alpha = 0.2
auto_alpha = True
alpha_lr = 3e-4
start_timesteps = 100
epoch = 5

### IMPORTANT!!
step_per_epoch = 100
step_per_collect = 10
update_per_step = 0.1
######

n_step = 1 #### Not very clear what it is --> see sppining bla bla
batch_size = 256
training_num = 1
test_num = 1
log_dir = "log"
render_flag = 0.
device = "cpu"
resume_path = None
logger_id = "tensorboard"
wandb_project = "mujoco.benchmark"
watch = False


def ivan_sac():

    env, train_envs, test_envs = make_mujoco_env(
        task, seed, training_num, test_num, obs_norm=False, render="human",
    )



    state_shape = env.observation_space.shape or env.observation_space.n
    action_shape = env.action_space.shape or env.action_space.n

    max_action = env.action_space.high[0]
    print("Observations shape:", state_shape)
    print("Actions shape:", action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))


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

    if auto_alpha:
        target_entropy = -np.prod(env.action_space.shape)
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
        alpha = (target_entropy, log_alpha, alpha_optim)

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
    if resume_path:
        policy.load_state_dict(torch.load(resume_path, map_location=device))
        print("Loaded agent from: ", resume_path)

    # collector
    if training_num > 1:
        buffer = VectorReplayBuffer(buffer_size, len(train_envs))
    else:
        buffer = ReplayBuffer(buffer_size)
    train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)
    test_collector = Collector(policy, test_envs)
    train_collector.collect(n_step=start_timesteps, random=True)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    algo_name = "sac"
    log_name = os.path.join(task, algo_name, str(seed), now)
    log_path = os.path.join(log_dir, log_name)

    # logger
    writer = SummaryWriter(log_path)
    if logger_id == "tensorboard":
        logger = TensorboardLogger(writer)
    #else:  # wandb
    #    logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    #if not watch:
    # trainer

    for id_epoch in range(epoch):

        result = offpolicy_trainer(
            policy,
            train_collector,
            test_collector,
            id_epoch,
            step_per_epoch,
            step_per_collect,
            test_num,
            batch_size,
            save_best_fn=save_best_fn,
            logger=logger,
            update_per_step=update_per_step,
            test_in_train=False,
        )
        pprint.pprint(result)

        # Let's watch its performance after every epoch!
        policy.eval()
        test_envs.seed(seed)
        test_collector.reset()
        result = test_collector.collect(n_step=200, render=render_flag )
        #print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')


#if __name__ == "__main__":
ivan_sac()
