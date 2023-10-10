from enum import Enum
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
from examples.offline_RL_workshop.offline_policies.ppo_policy import ppo_default_config, create_ppo_policy_from_dict


class CallableEnum(Enum):
    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class PolicyType(str, Enum):
    offline = "offline"
    onpolicy = "onpolicy"
    offpolicy = "offpolicy"


class PolicyName(str, Enum):
    bcq_discrete = "bcq_discrete"
    cql_continuous = "cql_continuous"
    imitation_learning = "imitation_learning"
    bcq_continuous = "bcq_continuous"
    cql_discrete = "cql_discrete"
    dqn = "dqn"
    ppo = "ppo"


class DefaultPolicyConfigFactoryRegistry(CallableEnum):
    bcq_discrete = bcq_discrete_default_config
    cql_continuous = cql_continuous_default_config
    imitation_learning = il_default_config
    bcq_continuous = bcq_continuous_default_config
    cql_discrete = cql_discrete_default_config
    dqn = dqn_default_config
    ppo = ppo_default_config


class PolicyFactoryRegistry(CallableEnum):
    bcq_discrete = create_bcq_discrete_policy_from_dict
    cql_continuous = create_cql_continuous_policy_from_dict
    imitation_learning = create_il_policy_from_dict
    bcq_continuous = create_bcq_continuous_policy_from_dict
    cql_discrete = create_cql_discrete_policy_from_dict
    dqn = create_dqn_policy_from_dict
    ppo = create_ppo_policy_from_dict

