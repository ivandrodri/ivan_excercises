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
POLICY_NAME = PolicyName.bcq_continuous

NUM_EPOCHS = 140
BATCH_SIZE = 256
UPDATE_PER_EPOCH = 50

NUMBER_TEST_ENVS = 5
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
    update_per_epoch=UPDATE_PER_EPOCH,
    restore_training=False,
)
