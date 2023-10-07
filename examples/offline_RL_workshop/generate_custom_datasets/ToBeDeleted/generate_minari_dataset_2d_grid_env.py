# ToDo: almost exactly the same file as generate_minari_dataset_custom_grid.py

import cv2
import gymnasium as gym
import minari
import numpy as np
from minari import DataCollectorV0, StepDataCallback


NAME_ENV = 'SimpleGrid-5x5-v1'
NAME_EXPERT_DATA = "simple_grid-gymnasium-data-v1"
NUM_STEPS = 100000

#env = gym.make(NAME_ENV, render_mode='rgb_array_list')
env = gym.make(NAME_ENV)
dataset_id = NAME_EXPERT_DATA


print(f"Observation space: {env.observation_space}")

local_datasets = minari.list_local_datasets()
# Delete dataset if in local folder offline/.minari/datasets
if dataset_id in local_datasets:
    minari.delete_dataset(dataset_id)


class CustomSubsetStepDataCallback(StepDataCallback):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        #del step_data["observations"]["achieved_goal"]
        return step_data


env = DataCollectorV0(
    env,
    step_data_callback=CustomSubsetStepDataCallback,
    record_infos=False,
)


#options = {"start_loc":(0,0), "goal_loc":(3,3)}
states, _ = env.reset()
#done = env.unwrapped.done

obstacle_xy = (2,2)#env.goal_xy
distance_to_goal=1

for i in range(NUM_STEPS):

    # Here add behavior policy
    action = env.action_space.sample()
    #action = custom_policy_simple_grid(env.action_space, env.agent_xy, obstacle_xy, distance_to_goal=distance_to_goal)
    #action = random_behavior_policy(env.action_space)

    if not env.discrete_action:
        max_index = int(np.argmax(action))
        # Create a one-hot encoded array
        one_hot_encoded = np.zeros_like(action)
        one_hot_encoded[max_index] = 1
        action = one_hot_encoded

    next_state, reward, done, time_out, info = env.step(action)

    if env.render_mode == 'rgb_array_list':
        frames = env.render()

        height, width, _ = frames[0].shape

        if i == 0:
            # Create a window to display the frames
            cv2.namedWindow('Video', cv2.WINDOW_NORMAL)  # Create a resizable window
            cv2.resizeWindow('Video', width, height)  # Set the window size to match frame dimensions

        cv2.imshow('Video', frames[0])
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break

    state = next_state

    if done or time_out:
        env.reset()


cv2.destroyAllWindows()

dataset = minari.create_dataset_from_collector_env(dataset_id=NAME_EXPERT_DATA, collector_env=env)

data = minari.load_dataset(NAME_EXPERT_DATA)
print("number of episodes collected: ",len(data))
for elem in data:
    #print(elem.actions)
    #print(elem.observations)
    #print(elem.terminations)
    #print(elem.rewards)
    reward=0.0
    for i, rew in enumerate(elem.rewards):
        reward+=rew*0.99**i
    print(reward)


