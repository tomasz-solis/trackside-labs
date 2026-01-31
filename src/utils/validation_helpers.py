"""Input validation utilities for critical functions."""

import logging
from typing import Any, List, Union

logger = logging.getLogger(__name__)


def validate_range(value: float, name: str, min_val: float, max_val: float) -> None:
    """
    Validate value is within range [min_val, max_val].

    Args:
        value: The value to validate
        name: Parameter name for error message
        min_val: Minimum allowed value (inclusive)
        max_val: Maximum allowed value (inclusive)

    Raises:
        ValueError: If value is outside the range
    """
    if not (min_val <= value <= max_val):
        raise ValueError(f"{name} must be between {min_val} and {max_val}, got {value}")
    logger.debug(f"Validated {name}={value} is in range [{min_val}, {max_val}]")


def validate_positive_int(value: int, name: str, min_val: int = 1) -> None:
    """
    Validate value is a positive integer >= min_val.

    Args:
        value: The value to validate
        name: Parameter name for error message
        min_val: Minimum allowed value (default 1)

    Raises:
        ValueError: If value is not an integer or is less than min_val
    """
    if not isinstance(value, int) or value < min_val:
        raise ValueError(f"{name} must be an integer >= {min_val}, got {value}")
    logger.debug(f"Validated {name}={value} is positive integer >= {min_val}")


def validate_enum(value: str, name: str, valid_values: List[str]) -> None:
    """
    Validate value is one of the valid enum values.

    Args:
        value: The value to validate
        name: Parameter name for error message
        valid_values: List of allowed values

    Raises:
        ValueError: If value is not in valid_values
    """
    if value not in valid_values:
        raise ValueError(f"{name} must be one of {valid_values}, got '{value}'")
    logger.debug(f"Validated {name}='{value}' is in {valid_values}")


def validate_position(
    value: Union[int, float], name: str, min_pos: int = 1, max_pos: int = 20
) -> None:
    """
    Validate grid or race position.

    Args:
        value: The position to validate (int or float that will be converted to int)
        name: Parameter name for error message
        min_pos: Minimum position (default 1)
        max_pos: Maximum position (default 20)

    Raises:
        ValueError: If position is outside valid range
    """
    # Convert to int if it's a float
    if isinstance(value, float):
        value = int(value)

    if not isinstance(value, int) or not (min_pos <= value <= max_pos):
        raise ValueError(f"{name} must be between {min_pos} and {max_pos}, got {value}")
    logger.debug(f"Validated {name}={value} is valid position")


def validate_year(
    value: int, name: str = "year", min_year: int = 2020, max_year: int = 2030
) -> None:
    """
    Validate year is within valid range.

    Args:
        value: Year to validate
        name: Parameter name for error message (default "year")
        min_year: Minimum allowed year (default 2020)
        max_year: Maximum allowed year (default 2030)

    Raises:
        ValueError: If year is outside valid range
    """
    if not isinstance(value, int) or not (min_year <= value <= max_year):
        raise ValueError(f"{name} must be between {min_year} and {max_year}, got {value}")
    logger.debug(f"Validated {name}={value} is in range [{min_year}, {max_year}]")
