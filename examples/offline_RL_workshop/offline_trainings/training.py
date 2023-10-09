from examples.offline_RL_workshop.custom_envs.custom_envs_registration import RenderMode
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyName
from examples.offline_RL_workshop.offline_trainings.offline_training import offline_training
from examples.offline_RL_workshop.offline_trainings.policy_config_data_class import TrainedPolicyConfig

NAME_EXPERT_DATA = "relocate-cloned-v1"
#"Grid_2D_8x8_discrete-combined_data_set-V0"
#"Grid_2D_8x8_discrete-data_obst_middle_8x8_start_0_0_target_7_7-v0"
#"Grid_2D_8x8_discrete-data_obst_middle_8x8_start_0_0_target_7_7-v0"
#"Grid_2D_4x4_discrete-data_obst_free_4x4_start_0_0_target_3_3-v0"
#"Grid_2D_8x8_discrete-data_obst_free_8x8_start_0_0_target_7_7-v0"
#"Ant-v2-data-v0"
POLICY_NAME = PolicyName.cql_continuous

NUM_EPOCHS = 140
BATCH_SIZE = 64
UPDATE_PER_EPOCH = 200

NUMBER_TEST_ENVS = 1
EXPLORATION_NOISE = True
SEED = None #1626


offline_policy_config = TrainedPolicyConfig(
    name_expert_data=NAME_EXPERT_DATA,
    policy_name=POLICY_NAME,
    render_mode=RenderMode.RGB_ARRAY_LIST,
    device="cpu"
)

offline_training(
    offline_policy_config=offline_policy_config,
    num_epochs = NUM_EPOCHS,
    number_test_envs=10,
)



'''
# Test Policy

policy_name = "policy.pth"

register_grid_envs()
env_name = offline_policy_config.minari_dataset_config.env_name
render_mode = offline_policy_config.render_mode
env_config = offline_policy_config.minari_dataset_config.initial_config_2d_grid_env

env = InitialConfigEnvWrapper(gym.make(env_name, render_mode=render_mode),
                              env_config=env_config)

# Policy restoration
exploraton_noise = True
policy_type = offline_policy_config.policy_name
policy_config = offline_policy_config.policy_config
policy = PolicyRestorationConfigFactoryRegistry.__dict__[policy_type]\
    (
        policy_config=policy_config,
        action_space=env.action_space,
        observation_space=env.observation_space
    )

name_expert_data = offline_policy_config.name_expert_data
log_name = os.path.join(name_expert_data, policy_type)
LOG_DIR = "../trained_models_data"
log_path = os.path.join(LOG_DIR, log_name)
policy.load_state_dict(torch.load(os.path.join(log_path, policy_name), map_location="cpu"))
final_collector = Collector(policy, env, exploration_noise=exploraton_noise)
final_collector.collect(n_episode=20, render=1 / 35)

'''
