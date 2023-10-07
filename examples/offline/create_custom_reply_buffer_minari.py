import minari
import gymnasium as gym
import numpy as np
import torch
from d4rl.pointmaze.waypoint_controller import WaypointController
from minari import DataCollectorV0, StepDataCallback

from examples.atari.atari_network import DQN
from examples.atari.atari_wrapper import make_atari_env, WarpFrame, FrameStack
from tianshou.data import Collector
from tianshou.policy import BasePolicy, DQNPolicy, ICMPolicy
from tianshou.utils.net.discrete import IntrinsicCuriosityModule

dataset_name = "PongNoFrameskip-v4-ivan"

# Check if dataset already exist and load to add more data
if dataset_name in minari.list_local_datasets():
    dataset = minari.load_dataset(dataset_name)
else:
    dataset = None

# continuing task => the episode doesn't terminate or truncate when reaching a goal
# it will generate a new target. For this reason we set the maximum episode steps to
# the desired size of our Minari dataset (evade truncation due to time limit)

task = "PongNoFrameskip-v4"
seed = 0
training_num = 1
test_num = 1
scale_obs = 0
frames_stack = 4

'''
env, train_envs, test_envs = make_atari_env(
        "PongNoFrameskip-v4",
        seed,
        training_num,
        test_num,
        scale=scale_obs,
        frame_stack=frames_stack,
    )
'''
#env = gym.make('PongNoFrameskip-v4')


# Data collector wrapper to save temporary data while stepping. Characteristics:
#   * Custom StepDataCallback to add extra state information to 'infos' and divide dataset in different episodes by overridng
#     truncation value to True when target is reached
#   * Record the 'info' value of every step
#collector_env = DataCollectorV0(env, record_infos=True)

#obs, _ = collector_env.reset(seed=123)


frame_stack = 4
inf_env = gym.make("PongNoFrameskip", render_mode="human")
#inf_env = gym.make("PongNoFrameskip")
inf_env = WarpFrame(inf_env)
inf_env = FrameStack(inf_env, frame_stack)

class PointMazeStepDataCallback(StepDataCallback):
    """Add environment state information to 'infos'.

    Also, since the environment generates a new target every time it reaches a goal, the environment is
    never terminated or truncated. This callback overrides the truncation value to True when the step
    returns a True 'succes' key in 'infos'. This way we can divide the Minari dataset into different trajectories.
    """
    def __call__(self, env, obs, info, action=None, rew=None, terminated=None, truncated=None):

        step_data = super().__call__(env, obs, info, action, rew, terminated, truncated)

        step_data["truncations"] = False
        step_data["terminations"] = False
        if step_data["rewards"]==-1.0:
            step_data["terminations"]=True

        return step_data

class CustomStepDataCallback(StepDataCallback):
    def __call__(self, env, **kwargs):
        step_data = super().__call__(env, **kwargs)
        #del step_data["observations"]["achieved_goal"]
        return step_data


#obs, info = inf_env.reset()
#print(obs)


collector_inf_env = DataCollectorV0(inf_env, step_data_callback=PointMazeStepDataCallback, record_infos=True)


## Load DQN
device = "cpu"
state_shape = inf_env.observation_space.shape or inf_env.observation_space.n
action_shape = inf_env.action_space.shape or inf_env.action_space.n
lr = 0.0001
icm_lr_scale = 0.0
icm_reward_scale = 0.01
icm_forward_loss_weight = 0.2
gamma = 0.99
n_step = 3
target_update_freq=500

print("Observations shape:", state_shape)
print("Actions shape:", action_shape)
net = DQN(*state_shape, action_shape, device).to(device)
optim = torch.optim.Adam(net.parameters(), lr=lr)
# define policy
policy = DQNPolicy(
    net,
    optim,
    gamma,
    n_step,
    target_update_freq=target_update_freq
)
if icm_lr_scale > 0:
    feature_net = DQN(
        *state_shape, action_shape, device, features_only=True
    )
    action_dim = np.prod(action_shape)
    feature_dim = feature_net.output_dim
    icm_net = IntrinsicCuriosityModule(
        feature_net.net,
        feature_dim,
        action_dim,
        hidden_sizes=[512],
        device=device
    )
    icm_optim = torch.optim.Adam(icm_net.parameters(), lr=lr)
    policy = ICMPolicy(
        policy, icm_net, icm_optim, icm_lr_scale, icm_reward_scale,
        icm_forward_loss_weight
    ).to(device)


resume_path = "/home/ivan/Documents/GIT_PROJECTS/Tianshou/tianshou/log/PongNoFrameskip-v4/dqn/0/230906-142843/policy.pth"
policy.load_state_dict(torch.load(resume_path, map_location=device))




# Define tianshou collector -- get initial obs -- do loop

#buffer_size = 10
#buffer = VectorReplayBuffer(
#    buffer_size,
#    buffer_num=len(train_envs),
#    ignore_obs_next=True,
#    save_only_last_obs=True,
#    stack_num=args.frames_stack
#)
# collector
#train_collector = Collector(policy, train_envs, buffer, exploration_noise=True)

#test_collector = Collector(policy, inf_env)
#test_collector.reset()
#test_collector.
#test_collector.reset_env()

observation, info = collector_inf_env.reset(seed=42)

print("ääääääää ", observation)

#action = policy.model(observation.reshape(1,4,84,84))
#print(action[0])

#action = torch.argmax(action[0])
#print(action)

dataset = None
for _ in range(10):
    #action = int(torch.argmax(policy.model(observation.reshape(1,4,84,84))[0]))
    obs, rew, truncation, termination, info = collector_inf_env.step(1)
    #print(rew, truncation, termination)
    #action = inf_env.action_space.sample()  # this is where you would insert your policy
    #observation, reward, terminated, truncated, info = inf_env.step(action)
    #observation = observation.reshape(1, 4, 84, 84)

    if dataset is None:
        dataset = minari.create_dataset_from_collector_env(collector_env=collector_inf_env,
                                                           dataset_id=dataset_name,
                                                           algorithm_name="DQNIteration",)
                                                          # code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/docs/tutorials/dataset_creation/point_maze_dataset.py",
                                                          # author="Rodrigo Perez-Vicente",
                                                          # author_email="rperezvicente@farama.org")
    else:
        # Update local Minari dataset every 200000 steps.
        # This works as a checkpoint to not lose the already collected data
        dataset.update_dataset_from_collector_env(collector_inf_env)

    #dataset = minari.create_dataset_from_collector_env(collector_env=collector_inf_env,
    #                                                   dataset_id=dataset_name,
    #                                                   algorithm_name="DQNIteration",
    #                                                   #code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/docs/tutorials/dataset_creation/point_maze_dataset.py",
    #                                                   #author="Rodrigo Perez-Vicente",
    #                                                   #author_email="rperezvicente@farama.org"
    #                                                    )


#result = test_collector.collect(n_step=5000)
    #result = test_collector.collect(n_episode=4)
#print("########################### ", result["rew"])

    #if terminated or truncated:
    #  observation, info = inf_env.reset()



#for n_step in range(1, int(4)):
#    action = inf_env.action_space.sample()
#    observation, reward, terminated, truncated, info  = inf_env.step(action)
#    inf_env.render()
    #print(observation)
    #action = policy.compute_action(obs)
    #result = test_collector.collect(n_step=1)
    #print(result)
    #print("################################")
    # Add some noise to each step action
    #action += np.random.randn(*action.shape)*0.5
    #action = np.clip(action, env.action_space.low, env.action_space.high, dtype=np.float32)

    #obs, rew, terminated, truncated, info = collector_env.step(action)
    #if (n_step + 1) % 200000 == 0:
    #    print('STEPS RECORDED:')
    #    print(n_step)
    #    if dataset is None:
    #        dataset = minari.create_dataset_from_collector_env(collector_env=collector_env,
    #                                                           dataset_name=dataset_name,
    #                                                           algorithm_name="QIteration",
    #                                                           code_permalink="https://github.com/Farama-Foundation/Minari/blob/main/docs/tutorials/dataset_creation/point_maze_dataset.py",
    #                                                           author="Rodrigo Perez-Vicente",
    #                                                           author_email="rperezvicente@farama.org")
    #    else:
    #        # Update local Minari dataset every 200000 steps.
    #        # This works as a checkpoint to not lose the already collected data
    #        dataset.update_dataset_from_collector_env(collector_env)
            
