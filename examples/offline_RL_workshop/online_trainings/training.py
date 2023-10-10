from examples.offline_RL_workshop.custom_envs.custom_envs_registration import RenderMode
from examples.offline_RL_workshop.offline_policies.policy_registry import PolicyName, PolicyType
from examples.offline_RL_workshop.offline_trainings.policy_config_data_class import TrainedPolicyConfig
from examples.offline_RL_workshop.online_trainings.online_training import online_training

NAME_EXPERT_DATA = "relocate-cloned-v1"
#"Grid_2D_8x8_discrete-combined_data_set-V0"
#"Grid_2D_8x8_discrete-data_obst_middle_8x8_start_0_0_target_7_7-v0"
#"Grid_2D_8x8_discrete-data_obst_middle_8x8_start_0_0_target_7_7-v0"
#"Grid_2D_4x4_discrete-data_obst_free_4x4_start_0_0_target_3_3-v0"
#"Grid_2D_8x8_discrete-data_obst_free_8x8_start_0_0_target_7_7-v0"
#"Ant-v2-data-v0"
POLICY_NAME = PolicyName.ppo
POLICY_TYPE = PolicyType.onpolicy

NUM_EPOCHS = 1
BATCH_SIZE = 64
STEP_PER_EPOCH = 100000
STEP_PER_COLLECT = 10
#UPDATE_PER_EPOCH = 200

NUMBER_TRAINING_ENVS = 10
NUMBER_TEST_ENVS = 5
EXPLORATION_NOISE = True
SEED = None #1626


policy_config = TrainedPolicyConfig(
    name_expert_data=NAME_EXPERT_DATA,
    policy_name=POLICY_NAME,
    render_mode=RenderMode.RGB_ARRAY_LIST,
    device="cpu"
)


if POLICY_NAME==PolicyName.ppo:
    policy_config.policy_config["lr_decay"]={
        "step_per_epoch": STEP_PER_EPOCH,
        "step_per_collect": STEP_PER_COLLECT,
        "epoch": NUM_EPOCHS,
    }


online_training(
    trained_policy_config=policy_config,
    policy_type=POLICY_TYPE,
    num_epochs=NUM_EPOCHS,
    batch_size=BATCH_SIZE,
    number_train_envs=NUMBER_TRAINING_ENVS,
    step_per_epoch=STEP_PER_EPOCH,
    step_per_collect=STEP_PER_COLLECT,
    number_test_envs=NUMBER_TEST_ENVS,
    restore_training=False,
)
