import warnings

import gymnasium as gym
#import gym

from tianshou.env import ShmemVectorEnv, VectorEnvNormObs

#try:
#    import envpool
#except ImportError:
#    envpool = None

envpool = None

def make_mujoco_env(task, seed, training_num, test_num, obs_norm, render=None):
    """Wrapper function for Mujoco env.

    If EnvPool is installed, it will automatically switch to EnvPool's Mujoco env.

    :return: a tuple of (single env, training envs, test envs).
    """
    if envpool is not None:
        train_envs = env = envpool.make_gymnasium(
            task, num_envs=training_num, seed=seed
        )
        test_envs = envpool.make_gymnasium(task, num_envs=test_num, seed=seed)
    else:
        warnings.warn(
            "Recommend using envpool (pip install envpool) "
            "to run Mujoco environments more efficiently."
        )

        env = gym.make(task)

        train_envs = ShmemVectorEnv(
            [lambda: gym.make(task) for _ in range(training_num)]
        )

        # ToDo : This should be change as it will collect render info and slow down trainings.
        #   Best option could be to create in main script a new vector environment only for rendering.
        render_mode = render
        test_envs = ShmemVectorEnv([lambda: gym.make(task, render_mode=render_mode) for _ in range(test_num)])


        #env.seed(seed)
        #train_envs.seed(seed)
        #test_envs.seed(seed)

        _, _ = env.reset(seed=0)
        _, _ = train_envs.reset(seed=0)
        _, _ = test_envs.reset(seed=0)

    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs
