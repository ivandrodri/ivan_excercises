import os
from typing import Tuple, Dict

import d4rl
import gymnasium as gym
import h5py
import minari
import numpy as np
from minari import EpisodeData
from minari.storage import get_dataset_path

from tianshou.data import ReplayBuffer, VectorReplayBuffer
from tianshou.utils import RunningMeanStd


def load_buffer_d4rl(expert_data_task: str) -> ReplayBuffer:
    import gym
    dataset = d4rl.qlearning_dataset(gym.make(expert_data_task))
    replay_buffer = ReplayBuffer.from_data(
    #replay_buffer = VectorReplayBuffer(buffer_num=1).from_data(
        obs=dataset["observations"],
        act=dataset["actions"],
        rew=dataset["rewards"],
        done=dataset["terminals"],
        obs_next=dataset["next_observations"],
        terminated=dataset["terminals"],
        truncated=np.zeros(len(dataset["terminals"]))
    )
    return replay_buffer


def _episode_data_lengths(episode:EpisodeData):
    obs = episode.observations["observation"] if isinstance(episode.observations, Dict) else \
        episode.observations
    lens_data = [obs[1:].shape[0], episode.actions.shape[0], episode.rewards.shape[0],
                 episode.truncations.shape[0], episode.terminations.shape[0]]

    return min(lens_data)


def load_buffer_minari(expert_data_task: str) -> ReplayBuffer:
    data_set_minari_paths = get_dataset_path("")
    data_set_minari_paths = os.path.join(data_set_minari_paths, "")
#    data_set_minari_paths = os.path.join(os.path.expanduser("~"), ".minari", "datasets")
    data_set_expert_task_path = os.path.join(data_set_minari_paths, expert_data_task)

    if not os.path.exists(data_set_expert_task_path):
        minari.download_dataset(expert_data_task)

    dataset = minari.load_dataset(expert_data_task)

    print(f"Dataset {data_set_expert_task_path} downloaded. number of episodes: {len(dataset)}")

    observations_list = []
    actions_list = []
    rewards_list = []
    terminals_list = []
    truncations_list = []
    next_observations_list = []

    for i, episode in enumerate(dataset):

        # For some data the len of the episode len data (observations, actions, etc.) is not the same
        common_len = _episode_data_lengths(episode)

        obs = episode.observations["observation"] if isinstance(episode.observations, Dict) \
            else episode.observations

        next_observations_list.append(obs[1:][0:common_len])
        observations_list.append(obs[:-1][0:common_len])
        terminals_list.append(episode.terminations[0:common_len])
        truncations_list.append(episode.truncations[0:common_len])
        rewards_list.append(episode.rewards[0:common_len])
        actions_list.append(episode.actions[0:common_len])

    observations = np.concatenate(observations_list, axis=0)
    actions = np.concatenate(actions_list, axis=0)
    terminals = np.concatenate(terminals_list, axis=0)
    next_observations = np.concatenate(next_observations_list, axis=0)
    rewards = np.concatenate(rewards_list, axis=0)
    #truncations = np.concatenate(truncations_list, axis=0)

    replay_buffer = ReplayBuffer.from_data(
        obs=observations,
        act=actions,
        rew=rewards,
        done=terminals,
        obs_next=next_observations,
        terminated=terminals,
        truncated=np.zeros(len(terminals))
    )
    return replay_buffer


def load_buffer(buffer_path: str) -> ReplayBuffer:
    with h5py.File(buffer_path, "r") as dataset:
        buffer = ReplayBuffer.from_data(
            obs=dataset["observations"],
            act=dataset["actions"],
            rew=dataset["rewards"],
            done=dataset["terminals"],
            obs_next=dataset["next_observations"],
            terminated=dataset["terminals"],
            truncated=np.zeros(len(dataset["terminals"]))
        )
    return buffer


def normalize_all_obs_in_replay_buffer(
    replay_buffer: ReplayBuffer
) -> Tuple[ReplayBuffer, RunningMeanStd]:
    # compute obs mean and var
    obs_rms = RunningMeanStd()
    obs_rms.update(replay_buffer.obs)
    _eps = np.finfo(np.float32).eps.item()
    # normalize obs
    replay_buffer._meta["obs"] = (replay_buffer.obs -
                                  obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    replay_buffer._meta["obs_next"] = (replay_buffer.obs_next -
                                       obs_rms.mean) / np.sqrt(obs_rms.var + _eps)
    return replay_buffer, obs_rms
