"""
AUJ Platform Registry Package.

This package contains configuration mappings and registries for the platform.
"""

from .agent_indicator_mapping import (
    AGENT_MAPPINGS as AGENT_INDICATOR_MAPPING,
    get_agent_indicators,
    get_all_mapped_indicators,
    get_agent_specialization,
    get_agents_for_indicator,
    validate_agent_mapping,
    get_mapping_summary
)

__all__ = [
    "AGENT_INDICATOR_MAPPING",
    "get_agent_indicators",
    "get_all_mapped_indicators",
    "get_agent_specialization",
    "get_agents_for_indicator",
    "validate_agent_mapping",
    "get_mapping_summary"
]
