import os

import gymnasium as gym
import torch
from torch.utils.tensorboard import SummaryWriter
import tianshou as ts
from examples.offline_RL_workshop.utils import integer_to_one_hot, extract_dimension
from tianshou.data import Batch

TRAIN = False

NAME_ENV = "SimpleGrid-8x8-v0"#"ivan_1d_grid-gymnasium-v0" #'CartPole-v0'
ALGO_NAME = "DQN_online"

task = NAME_ENV
lr, epoch, batch_size = 1e-3, 100, 256
train_num, test_num = 10, 100
gamma, n_step, target_freq = 0.9, 3, 320
buffer_size = 20000
eps_train, eps_test = 0.01, 0.005
step_per_epoch, step_per_collect = 1000, 10

# For other loggers: https://tianshou.readthedocs.io/en/master/tutorials/logger.html

# you can also try with SubprocVectorEnv
train_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(train_num)])
test_envs = ts.env.DummyVectorEnv([lambda: gym.make(task) for _ in range(test_num)])


from tianshou.utils.net.common import Net
# you can define other net by following the API:
# https://tianshou.readthedocs.io/en/master/tutorials/dqn.html#build-the-network
env = gym.make(task)
state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n
net = Net(state_shape=state_shape, action_shape=action_shape, hidden_sizes=[128, 128, 128])
optim = torch.optim.Adam(net.parameters(), lr=lr)


policy = ts.policy.DQNPolicy(net, optim, gamma, n_step, target_update_freq=target_freq)
train_collector = ts.data.Collector(policy, train_envs, ts.data.VectorReplayBuffer(buffer_size, train_num), exploration_noise=True)
test_collector = ts.data.Collector(policy, test_envs, exploration_noise=True)  # because DQN uses epsilon-greedy method

log_name = os.path.join(NAME_ENV, ALGO_NAME)
LOG_DIR = "../log"
log_path = os.path.join(LOG_DIR, log_name)

def save_best_fn(policy):
    torch.save(policy.state_dict(), os.path.join(log_path, "policy.pth"))

if TRAIN:
    logger = ts.utils.TensorboardLogger(SummaryWriter(log_path))  # TensorBoard is supported!


    result = ts.trainer.offpolicy_trainer(
        policy, train_collector, test_collector, epoch, step_per_epoch, step_per_collect,
        test_num, batch_size, update_per_step=1 / step_per_collect,
        train_fn=lambda epoch, env_step: policy.set_eps(eps_train),
        test_fn=lambda epoch, env_step: policy.set_eps(eps_test),
        save_best_fn = save_best_fn,
        #stop_fn=lambda mean_rewards: mean_rewards >= env.spec.reward_threshold,
        logger=logger)
    print(f'Finished training! Use {result["duration"]}')
    #torch.save(policy.state_dict(), log_path)
    torch.save(policy.state_dict(), os.path.join(log_path, "policy_final.pth"))
else:
    env = gym.make(task, render_mode="rgb_array_list")
    policy.load_state_dict(torch.load(os.path.join(log_path,"policy.pth")))
    policy.eval()
    policy.set_eps(eps_train)

    state_shape = extract_dimension(env.observation_space)
    q_value_matrix = {int1: 0 for int1 in range(state_shape + 1)}

    for state_id in range(state_shape):
        ### Compute Q_values
        state = integer_to_one_hot(state_id, state_shape-1)
        tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
        policy_output = policy(tensor_state)
        q_values = policy_output.logits[0]
        q_value_matrix[state_id]=q_values

    #collector = ts.data.Collector(policy, env, exploration_noise=True)
    #collector.collect(n_episode=3000, render=1 / 35)


    t=0
    reward = 0
    gamma = 0.9
    num_steps = 13
    for i in range(num_steps):
        reward += (-0.5)*(gamma)**i

    print("Cum Rew. : ", reward)

    for key, value in q_value_matrix.items():
        print(f"{key}: {value}")



# [-1.7329, -0.7974, -1.6822, -0.7987]