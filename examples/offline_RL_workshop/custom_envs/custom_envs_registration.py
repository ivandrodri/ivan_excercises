from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, Optional, Union
from gymnasium.envs.registration import register as gymnasium_register

from examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.obstacles_2D_grid_register import ObstacleTypes

ENTRY_POINT_2D_GRID = "examples.offline_RL_workshop.custom_envs.custom_2d_grid_env.simple_grid:SimpleGridEnv"
ENTRY_POINT_1D_GYMNASIUM = "examples.offline_RL_workshop.custom_envs.custom_1d_grid_env:CustomGridEnvGymnasium"


class RenderMode(str, Enum):
    RGB_ARRAY_LIST = "rgb_array_list"
    RGB_ARRAY = "rgb_array"
    HUMAN = "human"
    NONE = None


class CustomEnv(str, Enum):
    HalfCheetah_v5 = "HalfCheetah-v5"
    Grid_2D_4x4_discrete = "Grid_2D_4x4_discrete"
    Grid_2D_4x4_continuous = "Grid_2D_4x4_continuous"
    Grid_2D_6x6_discrete = "Grid_2D_6x6_discrete"
    Grid_2D_6x6_continuous = "Grid_2D_6x6_continuous"
    Grid_2D_8x8_continuous = "Grid_2D_8x8_continuous"
    Grid_2D_8x8_discrete = "Grid_2D_8x8_discrete"
    # ToDo: Change this but maybe 1d grid should be deleted
    Grid_1D_1x10_discrete_V0 = "Grid_1D_1x10_discrete_V0"
    Grid_1D_1x10_continuous_V0 = "Grid_1D_1x10_continuous_V0"


@dataclass
class GridEnvConfig:
    env_name: str
    grid_1d_dim: Optional[int] = None
    obstacles: Optional[List[str]] = None
    render_mode: RenderMode = None
    discrete_action: bool = True
    max_episode_steps: int = 60

    def __post_init__(self):

        if self.obstacles is not None and self.grid_1d_dim is not None:
            raise ValueError("Provide either 'obstacles' or 'grid_dims', but not both")

        if self.obstacles:
            # Compute GRID_DIMS from OBSTACLES if obstacles are provided
            num_rows = len(self.obstacles)
            num_cols = len(self.obstacles[0]) if num_rows > 0 else 0
            if not(num_rows>0 and num_cols > 0):
                raise ValueError("To use obstacle maps The grid must be two dimensional!")


def register_custom_grid_env(grid_env_config: GridEnvConfig):

    if grid_env_config.grid_1d_dim is not None:
        gymnasium_register(
            id=grid_env_config.env_name,
            entry_point=ENTRY_POINT_1D_GYMNASIUM,
            kwargs={
                "grid_size": grid_env_config.grid_1d_dim,
                "max_episode_steps": grid_env_config.max_episode_steps,
                "discrete_action": grid_env_config.discrete_action,
                "render_mode": grid_env_config.render_mode,
            }
        )
    elif grid_env_config.obstacles is not None:
        gymnasium_register(
            id=grid_env_config.env_name,
            entry_point=ENTRY_POINT_2D_GRID,
            max_episode_steps=grid_env_config.max_episode_steps,
            kwargs={
                'obstacle_map': grid_env_config.obstacles,
                'discrete_action': grid_env_config.discrete_action,
            },
        )
    else:
        raise ValueError("For a 2d grid you must pass an obstacle map")


def register_HalfCheetah_v5_env(max_episode_steps=50):
    env_name = CustomEnv.HalfCheetah_v5
    entry_point = "gymnasium.envs.mujoco:HalfCheetahEnv"
    reward_threshold = 4800.0

    return gymnasium_register(
            id=env_name,
            entry_point=entry_point,
            max_episode_steps=max_episode_steps,
            reward_threshold=reward_threshold,
        )


# ToDo: Code duplication a lot!
def register_grid_envs():
    max_episode_steps = 5
    register_HalfCheetah_v5_env(max_episode_steps=max_episode_steps)

    obstacles = ObstacleTypes.obst_free_4x4.value

    CONFIG = {
        "env_name": CustomEnv.Grid_2D_4x4_discrete,
        "obstacles": obstacles,
        "render_mode": RenderMode.RGB_ARRAY_LIST,
        "discrete_action": True,
    }
    config = GridEnvConfig(**CONFIG)
    register_custom_grid_env(config)

    CONFIG = {
        "env_name": CustomEnv.Grid_2D_4x4_continuous,
        "obstacles": obstacles,
        "render_mode": RenderMode.RGB_ARRAY_LIST,
        "discrete_action": False,
    }
    config = GridEnvConfig(**CONFIG)
    register_custom_grid_env(config)

    obstacles = ObstacleTypes.obst_free_6x6.value
    CONFIG = {
        "env_name": CustomEnv.Grid_2D_6x6_discrete,
        "obstacles": obstacles,
        "render_mode": RenderMode.RGB_ARRAY_LIST,
        "discrete_action": True,
    }

    config = GridEnvConfig(**CONFIG)
    register_custom_grid_env(config)

    CONFIG = {
        "env_name": CustomEnv.Grid_2D_6x6_continuous,
        "obstacles": obstacles,
        "render_mode": RenderMode.RGB_ARRAY_LIST,
        "discrete_action": False,
    }

    config = GridEnvConfig(**CONFIG)
    register_custom_grid_env(config)

    obstacles = ObstacleTypes.obst_free_8x8.value
    CONFIG = {
        "env_name": CustomEnv.Grid_2D_8x8_discrete,
        "obstacles": obstacles,
        "render_mode": RenderMode.RGB_ARRAY_LIST,
        "discrete_action": True,
    }

    config = GridEnvConfig(**CONFIG)
    register_custom_grid_env(config)

    CONFIG = {
        "env_name": CustomEnv.Grid_2D_8x8_continuous,
        "obstacles": obstacles,
        "render_mode": RenderMode.RGB_ARRAY_LIST,
        "discrete_action": False,
    }

    config = GridEnvConfig(**CONFIG)
    register_custom_grid_env(config)

    CONFIG = {
        "env_name": CustomEnv.Grid_1D_1x10_continuous_V0,
        "grid_1d_dim": 10,
        "render_mode": RenderMode.HUMAN,
        "discrete_action": False,
    }

    config = GridEnvConfig(**CONFIG)
    register_custom_grid_env(config)

    CONFIG = {
        "env_name": CustomEnv.Grid_1D_1x10_discrete_V0,
        "grid_1d_dim": 10,
        "render_mode": RenderMode.HUMAN,
        "discrete_action": True,
    }

    config = GridEnvConfig(**CONFIG)
    register_custom_grid_env(config)



'''


def register_custom_env_v1():

    gymnasium_register(
        id='SimpleGrid-v0',
        entry_point=ENTRY_POINT_2D_GRID,
        max_episode_steps=200,
    )

    gymnasium_register(
        id='SimpleGrid-8x8-v0',
        entry_point=ENTRY_POINT_2D_GRID,
        max_episode_steps=200,
        kwargs={'obstacle_map': '8x8',
                'discrete_action': True},
    )

    gymnasium_register(
        id='SimpleGrid-8x8-v1',
        entry_point=ENTRY_POINT_2D_GRID,
        max_episode_steps=200,
        kwargs={'obstacle_map': '8x8',
                'discrete_action': False},
    )

    gymnasium_register(
        id='SimpleGrid-4x4-v0',
        entry_point=ENTRY_POINT_2D_GRID,
        max_episode_steps=200,
        kwargs={'obstacle_map': '4x4',
                'discrete_action': True},
    )

    gymnasium_register(
        id='SimpleGrid-4x4-v1',
        entry_point=ENTRY_POINT_2D_GRID,
        max_episode_steps=200,
        kwargs={'obstacle_map': '4x4',
                'discrete_action': False},
    )


    gymnasium_register(
        id='SimpleGrid-5x5-v0',
        entry_point=ENTRY_POINT_2D_GRID,
        max_episode_steps=200,
        kwargs={'obstacle_map': '5x5',
                'discrete_action': True},
    )

    gymnasium_register(
        id='SimpleGrid-5x5-v1',
        entry_point=ENTRY_POINT_2D_GRID,
        max_episode_steps=200,
        kwargs={'obstacle_map': '5x5',
                'discrete_action': False},
    )



    gymnasium_register(
        id="ivan_1d_grid-gymnasium-v0",
        entry_point=ENTRY_POINT_1D_GYMNASIUM,
        kwargs={
            "grid_size": GRID_SIZE,
            "initial_state": INITIAL_STATE,
            "target_state": TARGET_STATE,
            "discrete_action": True,
            "render_mode": 'human',
        }
    )

    gymnasium_register(
        id="ivan_1d_grid-gymnasium-v1",
        entry_point=ENTRY_POINT_1D_GYMNASIUM,
        kwargs={
            "grid_size": GRID_SIZE,
            "initial_state": INITIAL_STATE,
            "target_state": True,
            "discrete_action": False,
            "render_mode": 'human',
        }
    )


register_custom_env()

'''