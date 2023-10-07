from enum import Enum
from examples.offline_RL_workshop.behavior_policies.custom_1d_grid_policy import custom_1d_grid_policy
from examples.offline_RL_workshop.behavior_policies.custom_2d_grid_policy import \
    suboptimal_behavior_policy_2d_grid_discrete, suboptimal_behavior_policy_2d_grid_discrete_case_a, \
    suboptimal_behavior_policy_2d_grid_discrete_case_b, suboptimal_behavior_policy_ivan


class CallableEnum(Enum):
    """
    Make the enum value itself callable to the enum key.
    """

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class BehaviorPolicyRestorationConfigFactoryRegistry(CallableEnum):
    behavior_1d_custom = custom_1d_grid_policy
    random = lambda action_space: action_space.sample()
    behavior_suboptimal_2d_grid_discrete = suboptimal_behavior_policy_2d_grid_discrete
    behavior_suboptimal_2d_grid_discrete_case_a = suboptimal_behavior_policy_2d_grid_discrete_case_a
    behavior_suboptimal_2d_grid_discrete_case_b = suboptimal_behavior_policy_2d_grid_discrete_case_b
    suboptimal_behavior_policy_2d_ivan = suboptimal_behavior_policy_ivan

class BehaviorPolicyType(str, Enum):
    behavior_1d_custom = "behavior_1d_custom"
    random = "random"
    behavior_suboptimal_2d_grid_discrete = "behavior_suboptimal_2d_grid_discrete"
    behavior_suboptimal_2d_grid_discrete_case_a = "behavior_suboptimal_2d_grid_discrete_case_a"
    behavior_suboptimal_2d_grid_discrete_case_b = "behavior_suboptimal_2d_grid_discrete_case_b"
    suboptimal_behavior_policy_2d_ivan = "suboptimal_behavior_policy_2d_ivan"