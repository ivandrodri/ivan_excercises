import gymnasium as gym
import torch
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import register_grid_envs, CustomEnv
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyName
from examples.offline_RL_workshop.offline_trainings.restore_policy_model import restore_trained_offline_policy
from examples.offline_RL_workshop.utils import extract_dimension, get_max_episode_steps_env, \
    change_max_episode_steps_env
from tianshou.data import Batch
from itertools import accumulate
import matplotlib.pyplot as plt

from tianshou.policy import DiscreteBCQPolicy, BCQPolicy, CQLPolicy, DiscreteCQLPolicy, DQNPolicy, ImitationPolicy


def get_q_values(policy_model, tensor_obs):
    if isinstance(policy_model, DiscreteBCQPolicy):
        value = float(torch.max(policy_model(tensor_obs).q_value[0]))
    elif isinstance(policy_model, BCQPolicy):
        state = tensor_obs["obs"]
        value = float(policy.critic1(state, policy_model(tensor_obs).act)[0])
    elif isinstance(policy_model, CQLPolicy):
        value = float(torch.max(policy_model(tensor_obs).logits[0]))
    elif isinstance(policy_model, DQNPolicy):
        value = float(torch.max(policy_model(tensor_obs).logits[0]))
    elif isinstance(policy_model, DiscreteCQLPolicy):
        value = 0.0
    elif isinstance(policy_model, ImitationPolicy):
        value = 0.0
    else:
        raise ValueError("The policy")

    return value


def get_value_function_and_discounted_cumulative_reward(
        gym_env,
        policy,
        num_episodes=1,
        gamma=0.99
):
    def compute_cumulative_list_reversed(input_list):
        cumulative_list = list(reversed(list(accumulate(reversed(input_list)))))
        return cumulative_list

    state_shape = extract_dimension(gym_env.observation_space)
    max_episode_steps = get_max_episode_steps_env(gym_env)

    mean_cumulative_rewards = [0.0] * max_episode_steps
    mean_values = [0.0] * max_episode_steps

    for _ in range(num_episodes):
        done = False
        truncated = False
        state, _ = gym_env.reset()
        rewards = [0.0] * max_episode_steps
        values = [0.0] * max_episode_steps
        t = 0
        while not (done or truncated):
            tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
            policy_output = policy(tensor_state)

            if isinstance(gym_env.action_space, gym.spaces.Discrete):
                action = int(policy_output.act[0])
            else:
                if isinstance(policy_output.act[0], torch.Tensor):
                    action = policy_output.act[0].detach().numpy()
                else:
                    action = policy_output.act[0]

            value = get_q_values(policy, tensor_state)

            # action = gym_env.action_space.sample()
            next_state, reward, done, truncated, info = gym_env.step(action)

            # print(reward, reward*(gamma**t))

            rewards[t] = reward * (gamma ** t)
            # print(policy_output.logits)
            # print(reward, t, value, done, truncated)
            values[t] = value

            t += 1
            state = next_state

        cumulative_rewards = compute_cumulative_list_reversed(rewards)

        mean_cumulative_rewards = [(x + y) / num_episodes for x, y in zip(cumulative_rewards, mean_cumulative_rewards)]
        # mean_values = [(x + y)/num_steps_traj for x, y in zip(values, mean_values)]
        mean_values = values

    return mean_cumulative_rewards, mean_values


DATA_SET_NAME = "Grid_2D_6x6_discrete-V0_data" #"HalfCheetah-v5_data"  #
POLICY_NAME = PolicyName.bcq_discrete
NUM_EPISODES = 1
EXPLORATION_NOISE = True

policy, policy_config = restore_trained_offline_policy(data_set_name=DATA_SET_NAME, policy_name=POLICY_NAME)

register_grid_envs()
env = gym.make(policy_config["ENV_NAME"], render_mode=policy_config["RENDER_MODE"])
#change_max_episode_steps_env(env, new_max_episode_steps=50)

mean_cumulative_rew, mean_values = get_value_function_and_discounted_cumulative_reward(
    env,
    policy,
    num_episodes=NUM_EPISODES,
    gamma=0.99
)

plt.plot(mean_cumulative_rew)
plt.show()
plt.plot(mean_values)
plt.show()

#print(mean_cumulative_rew)
#print(mean_values)




