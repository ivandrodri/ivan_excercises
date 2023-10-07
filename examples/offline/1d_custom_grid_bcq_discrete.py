import gym
import minari
#import gymnasium as gym
import numpy as np
import torch

from examples.atari.atari_network import DQN
from examples.custom_envs.custom_grid_env_d4rl import custom_grid_env_registration



import torch
import torch.nn as nn
from typing import Callable, Tuple, Any, Dict, Sequence, Optional, Union

from examples.offline.utils import load_buffer_d4rl
from tianshou.data import Collector
from tianshou.env import SubprocVectorEnv
from tianshou.policy import DiscreteBCQPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.discrete import Actor


class DQNVector(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_shape: Sequence[int],
        device: Union[str, int, torch.device] = "cpu",
        features_only: bool = False,
        output_dim: Optional[int] = None,
        layer_init: Callable[[nn.Module], nn.Module] = lambda x: x,
    ) -> None:
        super().__init__()
        self.device = device
        self.net = nn.Sequential(
            layer_init(nn.Linear(input_dim, 64)),  # Adjust input_dim and hidden layers as needed
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            self.output_dim = np.prod(self.net(torch.zeros(1, input_dim)).shape[1:])
        if not features_only:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(self.output_dim, 128)),  # Adjust output_dim and hidden layers as needed
                nn.ReLU(inplace=True),
                layer_init(nn.Linear(128, np.prod(action_shape)))
            )
            self.output_dim = np.prod(action_shape)
        elif output_dim is not None:
            self.net = nn.Sequential(
                self.net, layer_init(nn.Linear(self.output_dim, output_dim)),
                nn.ReLU(inplace=True)
            )
            self.output_dim = output_dim

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Optional[Any] = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[torch.Tensor, Any]:
        r"""Mapping: s -> Q(s, \*)."""
        obs = torch.as_tensor(obs, device=self.device, dtype=torch.float32)
        return self.net(obs), state












#####################

#minari.create_dataset_from_collector_env()




custom_grid_env_registration()
env = gym.make("ivan_1d_grid-v0")
train_envs = gym.make("ivan_1d_grid-v0")

#test_envs = gym.make("ivan_1d_grid-v0")

test_num=1
test_envs = SubprocVectorEnv(
        [lambda: gym.make("ivan_1d_grid-v0") for _ in range(test_num)]
    )

#env.get_dataset()



'''

state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

# should be N_FRAMES x H x W
print("Observations shape:", state_shape)
print("Actions shape:", action_shape)

# seed
seed = 1626
np.random.seed(seed)
torch.manual_seed(seed)

# model
device = "cpu"
feature_net = DQNVector(
    state_shape, action_shape, device=device, features_only=True
).to(device)


hidden_sizes=[512]
policy_net = Actor(
    feature_net,
    action_shape,
    device=device,
    hidden_sizes=hidden_sizes,
    softmax_output=False,
).to(device)

imitation_net = Actor(
    feature_net,
    action_shape,
    device=device,
    hidden_sizes=hidden_sizes,
    softmax_output=False,
).to(device)

actor_critic = ActorCritic(policy_net, imitation_net)

lr = 6.25e-5
optim = torch.optim.Adam(actor_critic.parameters(), lr=lr)

# define policy
gamma = 0.99
n_step = 1
target_update_freq = 8000
eps_test = 0.001
unlikely_action_threshold = 0.3
imitation_logits_penalty = 0.01

policy = DiscreteBCQPolicy(
    policy_net, imitation_net, optim, gamma, n_step,
    target_update_freq, eps_test, unlikely_action_threshold,
    imitation_logits_penalty
)


NAME_ENV = "ivan_1d_grid-v0"
NAME_EXPERT_DATA="ivan_1d_grid-v0"
D4RL = True

buffer = load_buffer_d4rl(NAME_EXPERT_DATA)

test_collector = Collector(policy, test_envs, exploration_noise=True)

'''

'''

    # load a previous policy
    
    
'''