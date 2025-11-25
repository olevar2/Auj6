"""
Coordination system exceptions.

This module contains all exception classes used across the coordination system.
"""


class CoordinationError(Exception):
    """Coordination related error."""
    pass


class ValidationError(Exception):
    """Validation related error."""
    pass


class AgentError(Exception):
    """Agent related error."""
    pass
