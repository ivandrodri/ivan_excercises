import sys
#sys.path.append('/tianshou/')

from typing import Union, Callable, Dict

import cv2
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from examples.offline_RL_workshop.behavior_policies.behavior_policy_registry import BehaviorPolicyType, \
    BehaviorPolicyRestorationConfigFactoryRegistry
from examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import ObstacleTypes
from examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.simple_grid import SimpleGridEnv
from examples.offline_RL_workshop.custom_envs.custom_envs_registration import register_grid_envs, RenderMode, CustomEnv
from examples.offline_RL_workshop.custom_envs.utils import is_instance_of_SimpleGridEnv, InitialConfigEnvWrapper, \
    Grid2DInitialConfig


def render_custom_policy_simple_grid(
        env_name: CustomEnv,
        render_mode: RenderMode,
        env_2D_grid_initial_config: Grid2DInitialConfig = None,
        behavior_policy_name: BehaviorPolicyType = None,
        behavior_policy: Callable[[np.ndarray], Union[int, np.ndarray]] = None,
        num_steps=100,
):
    if behavior_policy_name is None and behavior_policy is None:
        raise ValueError("Either behavior_policy_name or behavior_policy must be provided.")
    if behavior_policy_name is not None and behavior_policy is not None:
        raise ValueError(
            "Both behavior_policy_name and behavior_policy cannot be provided simultaneously.")
    # Only for the 2D grid environment.

    env = InitialConfigEnvWrapper(gym.make(env_name, render_mode=render_mode), env_config=env_2D_grid_initial_config)


    state, _ = env.reset()

    for _ in range(num_steps):

        if behavior_policy_name is not None:
            behavior_policy = BehaviorPolicyRestorationConfigFactoryRegistry.__dict__[behavior_policy_name]

            if behavior_policy_name == BehaviorPolicyType.random:
                action = env.action_space.sample()
            else:
                action = behavior_policy(state, env)
        else:
            action = behavior_policy(state, env)

        next_state, reward, done, time_out, info = env.step(action)
        num_steps += 1


        if render_mode==RenderMode.RGB_ARRAY_LIST:
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
            num_steps=0
        else:
            state = next_state


register_grid_envs()



env_initial_config = {
    "obstacles": [
        "00000000",
        "00000000",
        "00000000",
        "00001000",
        "00000000",
        "00000000",
        "00000000",
        "00000000",
    ],
    "initial_state": (0, 0 ),
    "target_state": (7, 0)
}

env_2D_grid_initial_config = Grid2DInitialConfig(
    obstacles=ObstacleTypes.obst_middle_8x8,
    initial_state=env_initial_config["initial_state"],
    target_state=env_initial_config["target_state"]
)

CONFIG = {
    "env_name": CustomEnv.Grid_2D_8x8_discrete,
    "render_mode": RenderMode.RGB_ARRAY_LIST,
    "behavior_policy": BehaviorPolicyType.behavior_suboptimal_2d_grid_discrete_case_a,
    "env_initial_config": env_initial_config,
}

render_custom_policy_simple_grid(env_name = CONFIG["env_name"],
                                 render_mode=CONFIG["render_mode"],
                                 behavior_policy_name=CONFIG["behavior_policy"],
                                 env_2D_grid_initial_config=env_2D_grid_initial_config,
                                 num_steps=1000)

