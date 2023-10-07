import minari
import gymnasium as gym

from examples.offline_RL_workshop.custom_envs.custom_envs_registration import RenderMode

DATASET_NAME = "relocate-cloned-v1"
ENV_NAME = "AdroitHandRelocate-v1"
RENDER_MODE = RenderMode.RGB_ARRAY_LIST

env = gym.make(ENV_NAME, render_mode=RENDER_MODE)
env.reset()

dataset = minari.load_dataset(DATASET_NAME)

for episode in dataset:

    observations = episode.observations
    actions = episode.actions
    rewards = episode.rewards

    initial_state = observations[0]
    initial_state_dict = {"initial_state_dict": {"qpos": initial_state}}

    state = env.get_env_state()
    state["hand_pos"] = ""
    state["hand_pos"] = ""

    #env.reset(options=initial_state_dict)


    bla="3"
    #env.set_env_state(initial_state_dict)

    #for i in range(len(actions)):
    #    print(actions[i])
    #    break
    break




'''

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
'''