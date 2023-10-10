import os
import gymnasium as gym
import numpy as np
import torch
from examples.offline.utils import load_buffer_minari
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import register_grid_envs
from examples.offline_RL_workshop.custom_envs.utils import InitialConfigEnvWrapper
from examples.offline_RL_workshop.offline_trainings.custom_tensorboard_callbacks import CustomSummaryWriter
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyFactoryRegistry
from examples.offline_RL_workshop.offline_trainings.policy_config_data_class import TrainedPolicyConfig, \
    get_trained_policy_path
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.trainer import offline_trainer
from tianshou.utils import TensorboardLogger


def offline_training(
        offline_policy_config: TrainedPolicyConfig,
        num_epochs=1,
        batch_size=64,
        update_per_epoch=20,
        number_test_envs=1,
        exploration_noise=True,
        restore_training=False,
        seed=None):

    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Load data
    name_expert_data = offline_policy_config.name_expert_data
    data_buffer = load_buffer_minari(name_expert_data)

    # Create environments
    register_grid_envs()
    env_name = offline_policy_config.minari_dataset_config.env_name
    render_mode = offline_policy_config.render_mode
    env_config = offline_policy_config.minari_dataset_config.initial_config_2d_grid_env

    env = InitialConfigEnvWrapper(gym.make(env_name, render_mode=render_mode),
                                  env_config=env_config)

    test_envs = SubprocVectorEnv(
        [lambda: InitialConfigEnvWrapper(gym.make(env_name), env_config=env_config)
         for _ in range(number_test_envs)]
    )

    # Policy restoration
    policy_name = offline_policy_config.policy_name
    policy_config = offline_policy_config.policy_config
    policy = PolicyFactoryRegistry.__dict__[policy_name]\
        (
            policy_config=policy_config,
            action_space=env.action_space,
            observation_space=env.observation_space
        )

    # Path to save models/config
    log_name = os.path.join(name_expert_data, policy_name)
    log_path = get_trained_policy_path(log_name)

    if restore_training:
        policy_path = os.path.join(log_path, 'policy.pth')
        policy.load_state_dict(torch.load(policy_path, map_location=offline_policy_config.device))
        print("Loaded policy from: ", policy_path)



    # Create collector for testing
    test_collector = Collector(policy, test_envs, exploration_noise=exploration_noise)


    def save_best_fn(policy):
        torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

    def stop_fn(mean_rewards):
        return False

    # Tensorboard writer
    custom_writer = CustomSummaryWriter(log_path, env)
    custom_writer.log_custom_info()
    logger = TensorboardLogger(custom_writer)

    # Training
    _ = offline_trainer(
            policy=policy,
            buffer=data_buffer,
            test_collector=test_collector,
            max_epoch=num_epochs,
            update_per_epoch=update_per_epoch,
            episode_per_test=number_test_envs,
            batch_size=batch_size,
            stop_fn=stop_fn,
            save_best_fn=save_best_fn,
            logger=logger,
        )


    # Save final policy
    torch.save(policy.state_dict(), os.path.join(log_path, 'policy_final.pth'))

    # Save config
    offline_policy_config.save_to_file()


