import gymnasium as gym
from examples.offline.utils import load_buffer_minari
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import register_grid_envs
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyType
from examples.offline_RL_workshop.offline_trainings.restore_policy_model import restore_trained_offline_policy
from examples.offline_RL_workshop.utils import state_action_histogram, compare_state_action_histograms
from examples.offline_RL_workshop.visualizations.utils import get_state_action_data_and_policy_grid_distributions

DATA_SET_NAME = "Grid_2D_6x6_discrete-V1_data" #"HalfCheetah-v5_data"#
POLICY_NAME = PolicyType.bcq_discrete
NUM_EPISODES = 1
EXPLORATION_NOISE = True

policy, policy_config = restore_trained_offline_policy(data_set_name=DATA_SET_NAME, policy_name=POLICY_NAME)

register_grid_envs()
env = gym.make(policy_config["ENV_NAME"], render_mode=policy_config["RENDER_MODE"])

data = load_buffer_minari(policy_config["NAME_EXPERT_DATA"])

print(f"number of elements: {len(data)}")

state_action_count_data, state_action_count_policy = \
    get_state_action_data_and_policy_grid_distributions(data, env, policy, num_episodes=NUM_EPISODES)
state_action_histogram(state_action_count_data)

state_action_histogram(state_action_count_policy)
compare_state_action_histograms(state_action_count_data, state_action_count_policy)


