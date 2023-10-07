# 0 - Create Enum with policy-types
from enum import Enum
from functools import partial

import gym

from examples.offline_RL_workshop.offline_policies.bcq_continuous_policy import create_bcq_continuous_policy_from_dict, \
    bcq_continuous_default_config
from examples.offline_RL_workshop.offline_policies.bcq_discrete_policy import create_bcq_discrete_policy_from_dict, \
    bcq_discrete_default_config
from examples.offline_RL_workshop.offline_policies.cql_continuous_policy import create_cql_continuous_policy_from_dict, \
    cql_continuous_default_config
from examples.offline_RL_workshop.offline_policies.cql_discrete_policy import create_cql_discrete_policy_from_dict, \
    cql_discrete_default_config
from examples.offline_RL_workshop.offline_policies.dqn_policy import create_dqn_policy_from_dict, dqn_default_config
from examples.offline_RL_workshop.offline_policies.il_policy import create_il_policy_from_dict, il_default_config


class CallableEnum(Enum):
    """
    Make the enum value itself callable to the enum key.
    """

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class PolicyType(str, Enum):
    bcq_discrete = "bcq_discrete"
    cql_continuous = "cql_continuous"
    imitation_learning = "imitation_learning"
    bcq_continuous = "bcq_continuous"
    cql_discrete = "cql_discrete"
    dqn = "dqn"
    #random = "random"


class DefaultPolicyConfigFactoryRegistry(CallableEnum):
    bcq_discrete = bcq_discrete_default_config
    cql_continuous = cql_continuous_default_config
    imitation_learning = il_default_config
    bcq_continuous = bcq_continuous_default_config
    cql_discrete = cql_discrete_default_config
    dqn = dqn_default_config


#def random_policy(action_space: gym.Space):
#    return action_space.sample()


class PolicyRestorationConfigFactoryRegistry(CallableEnum):
    bcq_discrete = create_bcq_discrete_policy_from_dict
    cql_continuous = create_cql_continuous_policy_from_dict
    imitation_learning = create_il_policy_from_dict
    bcq_continuous = create_bcq_continuous_policy_from_dict
    cql_discrete = create_cql_discrete_policy_from_dict
    dqn = create_dqn_policy_from_dict
    #random = random_policy



# 1 - Create registry Enum with policy name-paths-type(from 0)

# 2 - Create factory from enum-#1
'''
class PolicyFactory:
    def __init__(self, policy_dir):
        self.policy_dir = policy_dir

    def create_policy(self, policy_name):
        if policy_name == "policy1":

            # Define and return the policy (e.g., a custom neural network)
            policy = CustomPolicy()
        elif policy_name == "policy2":
            # Define and return another policy
            policy = AnotherPolicy()
        else:
            raise ValueError(f"Unknown policy name: {policy_name}")

        # Check if a model file exists for the policy and load it if available
        model_file_path = os.path.join(self.policy_dir, f"{policy_name}_model.pth")
        if os.path.exists(model_file_path):
            policy.load_state_dict(torch.load(model_file_path))
            print(f"Loaded model for {policy_name} from {model_file_path}")
        else:
            print(f"No model found for {policy_name}")

        return policy
'''