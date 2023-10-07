import cv2
import gymnasium as gym

from examples.offline_RL_workshop.behavior_policies.behavior_policy_registry import BehaviorPolicyType, \
    BehaviorPolicyRestorationConfigFactoryRegistry
from examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import ObstacleTypes
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import RenderMode, GridEnvConfig, \
    register_custom_grid_env, CustomEnv
from examples.offline_RL_workshop.custom_envs.utils import Grid2DInitialConfig, InitialConfigEnvWrapper

# One dimensional grid environment
CONFIG = {
    "env_name": "ivan_1d_grid-gymnasium-v0",
    "grid_1d_dim": 10,
    "render_mode": RenderMode.HUMAN,
    "discrete_action": False,
}

config = GridEnvConfig(**CONFIG)
register_custom_grid_env(config)

env = gym.make(config.env_name, render_mode=config.render_mode)
state, info = env.reset()

print(state)


# 2D dimensional grid environment
obstacles = ObstacleTypes.obst_free_8x8.value

CONFIG = {
    "env_name": "env_2D_grid_4x4",
    "render_mode": RenderMode.RGB_ARRAY_LIST,
    "discrete_action": True,
    "obstacles": obstacles,
}

config = GridEnvConfig(**CONFIG)
register_custom_grid_env(config)

env = gym.make(config.env_name, render_mode=config.render_mode)


initial_state = (0, 0)
target_state = (7, 0)
#new_obstacles = [
#    "0000",
#    "0000",
#    "0000",
#    "0000",
#]
#new_obstacles = ObstacleTypes(new_obstacles)

new_obstacles = ObstacleTypes.obst_free_8x8

grid2D_config = Grid2DInitialConfig(initial_state=initial_state, target_state=target_state,
                                    obstacles=new_obstacles)

#print(grid2D_config)


env = InitialConfigEnvWrapper(env, grid2D_config)



state, info = env.reset()

print(state)

num_steps = 1000
render_mode = RenderMode.RGB_ARRAY_LIST
behavior_policy_name = BehaviorPolicyType.suboptimal_behavior_policy_2d_ivan

for _ in range(num_steps):

    if behavior_policy_name is not None:
        behavior_policy = BehaviorPolicyRestorationConfigFactoryRegistry.__dict__[behavior_policy_name]

        if behavior_policy_name == BehaviorPolicyType.random:
            action = env.action_space.sample()
        else:
            action = behavior_policy(state, env)
    else:
        action = behavior_policy(state, env)

    #action = env.action_space.sample()

    next_state, reward, done, time_out, info = env.step(action)
    num_steps += 1

    if render_mode == RenderMode.RGB_ARRAY_LIST:
        rendered_data = env.render()
        frames = rendered_data[0]
        height, width, _ = frames.shape

        cv2.imshow('Video', frames)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            raise InterruptedError("You quited ('q') the iteration.")
    else:
        env.render()

    if done or time_out:
        state, _ = env.reset()
        num_steps = 0
    else:
        state = next_state

