#!/usr/bin/env python3

import argparse
import datetime
import os
import pprint
from typing import Dict

import numpy as np
import torch
from gym.spaces import Box
from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym

from examples.offline.utils import load_buffer_d4rl, load_buffer_minari
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import BCQPolicy
from tianshou.trainer import offline_trainer
from tianshou.utils import TensorboardLogger, WandbLogger
from tianshou.utils.net.common import MLP, Net
from tianshou.utils.net.continuous import VAE, Critic, Perturbation

#NAME_ENV = "Ant-v3"
#NAME_EXPERT_DATA="ant-medium-expert-v0"
#D4RL = True


#NAME_ENV = "HalfCheetah"
#NAME_EXPERT_DATA="halfcheetah-expert"
#D4RL = True

#NAME_ENV = "PointMaze_Medium-v3"
#NAME_EXPERT_DATA="pointmaze-medium-v1"
#D4RL = False

#NAME_ENV = "PointMaze_Medium-v3"
#NAME_EXPERT_DATA="pointmaze-umaze-v1"
#D4RL = False

#NAME_ENV = "AdroitHandPen-v1"
#NAME_EXPERT_DATA="pen-expert-v1"
#D4RL = False

#NAME_ENV = "PointMaze_Medium-v3"
#NAME_EXPERT_DATA="maze2d-umaze-v1"
#D4RL = True

NAME_ENV = "PointMaze_Medium-v3"
NAME_EXPERT_DATA="point-maze-subseted_Ivan_v0"
D4RL = False




def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default=NAME_ENV)
    parser.add_argument(
            "--expert-data-task", type=str, default=NAME_EXPERT_DATA
        )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--buffer-size", type=int, default=1000000)
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[256, 256])
    parser.add_argument("--actor-lr", type=float, default=1e-3)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--start-timesteps", type=int, default=10000)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--step-per-epoch", type=int, default=50)
    parser.add_argument("--n-step", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--test-num", type=int, default=2)
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--render", type=float, default=1 / 35)

    parser.add_argument("--vae-hidden-sizes", type=int, nargs="*", default=[512, 512])
    # default to 2 * action_dim
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--gamma", default=0.99)
    parser.add_argument("--tau", default=0.005)
    # Weighting for Clipped Double Q-learning in BCQ
    parser.add_argument("--lmbda", default=0.75)
    # Max perturbation hyper-parameter for BCQ
    parser.add_argument("--phi", default=0.05)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--resume-path", type=str, default=None)
    parser.add_argument("--resume-id", type=str, default=None)
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb"],
    )
    parser.add_argument("--wandb-project", type=str, default="offline_d4rl.benchmark")
    parser.add_argument(
        "--watch",
        default=False,
        action="store_true",
        help="watch the play of pre-trained policy only",
    )
    return parser.parse_args()


def custom_env_registration(max_episode_steps:int):
    id = "HalfCheetah-v5"
    entry_point = "gymnasium.envs.mujoco:HalfCheetahEnv"
    reward_threshold = 4800.0

    return gym.envs.register(
            id=id,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            reward_threshold=reward_threshold,
        )


def create_collectors():
    pass

def teto_bcq():

    # ToDo: hardcoded --> To be changed
    #custom_env_registration(max_episode_steps=50)

    args = get_args()
    env = gym.make(args.task, render_mode='human')


    #from examples.custom_env_d4rl import CustomGridEnv
    #env = CustomGridEnv(10, 2, 7)

    # ToDo: This must change dependig on observation type (Is this because gym/gymnasium incomp.)

    if isinstance(env.observation_space, gym.spaces.dict.Dict):
        args.state_shape = env.observation_space["observation"].shape
    elif isinstance(env.observation_space, gym.spaces.box.Box):
        args.state_shape = env.observation_space.shape or env.observation_space.n
    else:
        raise ValueError(f"the observation_space object must be of one of these types "
                         f"{Dict, Box} but a type {type(env.observation_space)} was given")
    #args.state_shape = env.observation_space["observation"].shape

    args.action_shape = env.action_space.shape or env.action_space.n
    args.max_action = env.action_space.high[0]  # float
    print("device:", args.device)
    print("Observations shape:", args.state_shape)
    print("Actions shape:", args.action_shape)
    print("Action range:", np.min(env.action_space.low), np.max(env.action_space.high))

    args.state_dim = args.state_shape[0]
    args.action_dim = args.action_shape[0]
    print("Max_action", args.max_action)

    # test_envs = gym.make(args.task)
    test_envs = SubprocVectorEnv(
        [lambda: gym.make(args.task) for _ in range(args.test_num)]
    )
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    test_envs.seed(args.seed)

    # model
    # perturbation network
    net_a = MLP(
        input_dim=args.state_dim + args.action_dim,
        output_dim=args.action_dim,
        hidden_sizes=args.hidden_sizes,
        device=args.device,
    )
    actor = Perturbation(
        net_a, max_action=args.max_action, device=args.device, phi=args.phi
    ).to(args.device)
    actor_optim = torch.optim.Adam(actor.parameters(), lr=args.actor_lr)

    net_c1 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    net_c2 = Net(
        args.state_shape,
        args.action_shape,
        hidden_sizes=args.hidden_sizes,
        concat=True,
        device=args.device,
    )
    critic1 = Critic(net_c1, device=args.device).to(args.device)
    critic1_optim = torch.optim.Adam(critic1.parameters(), lr=args.critic_lr)
    critic2 = Critic(net_c2, device=args.device).to(args.device)
    critic2_optim = torch.optim.Adam(critic2.parameters(), lr=args.critic_lr)

    # vae
    # output_dim = 0, so the last Module in the encoder is ReLU
    vae_encoder = MLP(
        input_dim=args.state_dim + args.action_dim,
        hidden_sizes=args.vae_hidden_sizes,
        device=args.device,
    )
    if not args.latent_dim:
        args.latent_dim = args.action_dim * 2
    vae_decoder = MLP(
        input_dim=args.state_dim + args.latent_dim,
        output_dim=args.action_dim,
        hidden_sizes=args.vae_hidden_sizes,
        device=args.device,
    )
    vae = VAE(
        vae_encoder,
        vae_decoder,
        hidden_dim=args.vae_hidden_sizes[-1],
        latent_dim=args.latent_dim,
        max_action=args.max_action,
        device=args.device,
    ).to(args.device)
    vae_optim = torch.optim.Adam(vae.parameters())

    policy = BCQPolicy(
        actor,
        actor_optim,
        critic1,
        critic1_optim,
        critic2,
        critic2_optim,
        vae,
        vae_optim,
        device=args.device,
        gamma=args.gamma,
        tau=args.tau,
        lmbda=args.lmbda,
    )

    # load a previous policy
    if args.resume_path:
        policy.load_state_dict(torch.load(args.resume_path, map_location=args.device))
        print("Loaded agent from: ", args.resume_path)

    # collector
    test_collector = Collector(policy, test_envs)

    # log
    now = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    args.algo_name = "bcq"
    log_name = os.path.join(args.task, args.algo_name, str(args.seed), now)
    log_path = os.path.join(args.logdir, log_name)

    # logger
    if args.logger == "OfflineTrainerwandb":
        logger = WandbLogger(
            save_interval=1,
            name=log_name.replace(os.path.sep, "__"),
            run_id=args.resume_id,
            config=args,
            project=args.wandb_project,
        )
    writer = SummaryWriter(log_path)
    writer.add_text("args", str(args))
    if args.logger == "tensorboard":
        logger = TensorboardLogger(writer)
    else:  # wandb
        logger.load(writer)

    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def watch():
        if args.resume_path is None:
            args.resume_path = os.path.join(log_path, "policy.pth")

        policy.load_state_dict(
            torch.load(args.resume_path, map_location=torch.device("cpu"))
        )
        policy.eval()
        collector = Collector(policy, env)
        collector.collect(n_episode=10, render=args.render)
        #env.reset()

    if not args.watch:

        # ToDo: This cannot be hardcoded. Decide if use D4RL or minari or both
        if D4RL:
            replay_buffer = load_buffer_d4rl(args.expert_data_task)
        else:
            replay_buffer = load_buffer_minari(args.expert_data_task)

        # trainer
        for id_epoch in range(args.epoch):
            result = offline_trainer(
                policy,
                replay_buffer,
                test_collector,
                id_epoch,
                args.step_per_epoch,
                args.test_num,
                args.batch_size,
                save_best_fn=save_best_fn,
                logger=logger,
            )
            pprint.pprint(result)

            # Let's watch its performance after every epoch!
            policy.eval()
            collector = Collector(policy, env)
            collector.collect(n_step=500, render=args.render)
            #collector.collect(n_step=500)

    else:
        watch()

#        result = offline_trainer(
#            policy,
#            replay_buffer,
#            test_collector,
#            args.epoch,
#            args.step_per_epoch,
#            args.test_num,
#            args.batch_size,
#            save_best_fn=save_best_fn,
#            logger=logger,
#        )
#        pprint.pprint(result)
#    else:
#        watch()


    # Let's watch its performance!
    #policy.eval()
    #test_envs.seed(args.seed)
    #test_collector.reset()
    #result = test_collector.collect(n_episode=args.test_num, render=args.render)
    #print(result)
    #print(f"Final reward: {result['rews'].mean()}, length: {result['lens'].mean()}")

    # ToDo: To avoid error with rendering closing
    #import glfw
    #glfw.terminate()

if __name__ == "__main__":
    teto_bcq()
