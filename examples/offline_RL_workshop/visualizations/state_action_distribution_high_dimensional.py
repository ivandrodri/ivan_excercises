import gymnasium as gym
import torch

from examples.offline.utils import load_buffer_minari
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import register_grid_envs
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyName
from examples.offline_RL_workshop.offline_trainings.restore_policy_model import restore_trained_offline_policy
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from examples.offline_RL_workshop.utils import extract_dimension, one_hot_to_integer
from tianshou.data import Batch


def tsne_plot(dataset):
    # Flatten the data points into 1D arrays
    flattened_data = [(state.flatten(), action.flatten()) for state, action in dataset]
    # Convert the flattened data into a NumPy array
    X = np.array([np.concatenate((state, action)) for state, action in flattened_data])
    # Apply t-SNE for dimensionality reduction (2D visualization)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    X_tsne = tsne.fit_transform(X)
    # Create a scatter plot to visualize the data in 2D
    plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], cmap=plt.cm.get_cmap("viridis", 1))
    plt.title("t-SNE Visualization of Dataset")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.show()


DATA_SET_NAME = "HalfCheetah-v5_data"#"Grid_2D_6x6_discrete-V0_data" #
POLICY_NAME = PolicyName.cql_continuous
NUM_EPISODES = 500
EXPLORATION_NOISE = True

policy, policy_config = restore_trained_offline_policy(data_set_name=DATA_SET_NAME, policy_name=POLICY_NAME)

register_grid_envs()
env = gym.make(policy_config["ENV_NAME"], render_mode=policy_config["RENDER_MODE"])

data = load_buffer_minari(policy_config["NAME_EXPERT_DATA"])
print(f"number of elements: {len(data)}")


# Collect data from dataset
state_action_count_data = []
for episode in data:
    state_action_count_data.append((episode.obs, episode.act))


# Collect data from policy

state_action_count_policy = []
state_shape = extract_dimension(env.observation_space)

for i in range(NUM_EPISODES):
    done = False
    truncated = False
    state, _ = env.reset()
    while not (done or truncated):

        tensor_state = Batch({"obs": state.reshape(1, state_shape), "info": {}})
        policy_output = policy(tensor_state)

        if isinstance(env.action_space, gym.spaces.Discrete):
            action = int(policy_output.act[0])
        else:
            if isinstance(policy_output.act[0], torch.Tensor):
                action = policy_output.act[0].detach().numpy()
            else:
                action = policy_output.act[0]

        state_action_count_policy.append((state, action))
        next_state, reward, done, truncated, info = env.step(action)
        state = next_state


tsne_plot(state_action_count_data)
tsne_plot(state_action_count_policy)


# Example dataset: list of pairs (state, action)
#dataset = [(np.random.rand(1, 10), np.random.rand(1, 6)) for _ in range(100)]
