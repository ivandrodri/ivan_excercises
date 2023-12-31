from typing import Dict, Any, Union, Optional, Callable, Tuple, Sequence

import numpy as np
import torch
from torch import nn

from examples.atari.atari_network import QRDQN, DQN
from examples.offline_RL_workshop.utils import extract_dimension
from tianshou.policy import DiscreteCQLPolicy
import gymnasium as gym

policy_config = {
    "lr": 0.0001,  # 6.25e-5
    "gamma": 0.99,
    "n_step": 10,
    "target_update_freq": 10,
    "num_quantiles": 200,
    "min_q_weight": 10.0,
    "device": "cpu",
}

def cql_discrete_default_config():
    return policy_config

def create_cql_discrete_policy_from_dict(policy_config:Dict[str,Any], action_space:gym.core.ActType,
                                         observation_space:gym.core.ObsType):

    observation_shape = extract_dimension(observation_space)
    action_shape = extract_dimension(action_space)


    # ToDo: Adapt to vector.
    class QRDQN_test(DQN):
        """Reference: Distributional Reinforcement Learning with Quantile \
        Regression.

        For advanced usage (how to customize the network), please refer to
        :ref:`build_the_network`.
        """

        def __init__(
                self,
                observation_shape: int,
                action_dim: int,
                num_quantiles: int = 200,
                device: Union[str, int, torch.device] = "cpu",
        ) -> None:
            self.action_num = action_dim#np.prod(action_shape)
            super().__init__(observation_shape, [self.action_num * num_quantiles], device)
            self.num_quantiles = num_quantiles

        def forward(
                self,
                obs: Union[np.ndarray, torch.Tensor],
                state: Optional[Any] = None,
                info: Dict[str, Any] = {},
        ) -> Tuple[torch.Tensor, Any]:
            r"""Mapping: x -> Z(x, \*)."""
            obs, state = super().forward(obs)
            obs = obs.view(-1, self.action_num, self.num_quantiles)
            return obs, state

    net = QRDQN_test(observation_shape, action_shape, policy_config["num_quantiles"], device=policy_config["device"])


    optim = torch.optim.Adam(net.parameters(), lr=policy_config["lr"])
    ## define policy
    policy = DiscreteCQLPolicy(
        net,
        optim,
        policy_config["gamma"],
        policy_config["num_quantiles"],
        policy_config["n_step"],
        policy_config["target_update_freq"],
        min_q_weight=policy_config["min_q_weight"],
    ).to(policy_config["device"])


    return policy




