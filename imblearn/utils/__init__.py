"""
The :mod:`imblearn.utils` module includes various utilities.
"""

from ._docstring import Substitution
from ._validation import (
    check_neighbors_object,
    check_sampling_strategy,
    check_target_type,
)
from ._weight_accounting import (
    ResamplingAudit,
    WeightContributions,
    WeightedResamplingResult,
    compose_weighted_results,
    validate_sample_weight,
)

__all__ = [
    "check_neighbors_object",
    "check_sampling_strategy",
    "check_target_type",
    "Substitution",
    "ResamplingAudit",
    "WeightContributions",
    "WeightedResamplingResult",
    "compose_weighted_results",
    "validate_sample_weight",
]
