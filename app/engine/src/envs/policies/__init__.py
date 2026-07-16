# from .base import Policy
# from .random_policy import RandomPolicy
from envs.policies.base import Policy
from envs.policies.random_policy import RandomPolicy
from envs.policies.registry import register_policy

__all__ = ["Policy", "RandomPolicy", "register_policy"]