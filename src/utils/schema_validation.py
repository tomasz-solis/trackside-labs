"""
JSON Schema validation module for F1 prediction system.

Provides schemas and validation functions for driver characteristics, team characteristics,
and track characteristics to prevent crashes from malformed JSON data.
"""

import logging
from typing import Dict, Any

try:
    from jsonschema import validate, ValidationError
except ImportError:
    # Fallback if jsonschema not installed
    validate = None
    ValidationError = Exception

logger = logging.getLogger(__name__)


# ============================================================================
# DRIVER CHARACTERISTICS SCHEMA
# ============================================================================
DRIVER_CHARACTERISTICS_SCHEMA = {
    "type": "object",
    "required": ["drivers"],
    "properties": {
        "year": {"type": "integer", "minimum": 2020, "maximum": 2030},
        "last_updated": {"type": "string"},
        "extraction_type": {"type": "string"},
        "description": {"type": "string"},
        "drivers": {
            "type": "object",
            "patternProperties": {
                "^[A-Z]{3}$": {  # 3-letter driver code (e.g., VER, NOR, HAM)
                    "type": "object",
                    "required": ["racecraft", "pace", "dnf_risk"],
                    "properties": {
                        "name": {"type": "string"},
                        "number": {"type": "integer"},
                        "teams": {"type": "array", "items": {"type": "string"}},
                        "experience": {
                            "type": "object",
                            "properties": {
                                "total_seasons": {"type": "integer", "minimum": 0},
                                "total_races": {"type": "integer", "minimum": 0},
                                "tier": {"type": "string"},
                            },
                        },
                        "pace": {
                            "type": "object",
                            "required": ["quali_pace", "race_pace"],
                            "properties": {
                                "quali_pace": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "quali_std": {"type": "number", "minimum": 0},
                                "race_pace": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "race_std": {"type": "number", "minimum": 0},
                                "confidence": {"type": "string"},
                            },
                        },
                        "racecraft": {
                            "type": "object",
                            "required": ["skill_score", "overtaking_skill"],
                            "properties": {
                                "skill_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "overtaking_skill": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "driver_type": {"type": "string"},
                                "interpretation": {"type": "string"},
                                "races_analyzed": {"type": "integer", "minimum": 0},
                                "total_dnfs": {"type": "integer", "minimum": 0},
                            },
                        },
                        "dnf_risk": {
                            "type": "object",
                            "required": ["dnf_rate"],
                            "properties": {
                                "dnf_rate": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "risk_level": {"type": "string"},
                                "total_races": {"type": "integer", "minimum": 0},
                                "total_dnfs": {"type": "integer", "minimum": 0},
                                "dnf_types": {"type": "object"},
                            },
                        },
                        "tire_management": {
                            "type": "object",
                            "properties": {
                                "skill": {"type": "number", "minimum": 0, "maximum": 1},
                                "skill_score": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 1,
                                },
                                "baseline": {"type": "string"},
                                "notes": {"type": "string"},
                            },
                        },
                    },
                }
            },
        },
    },
}


# ============================================================================
# TEAM CHARACTERISTICS SCHEMA
# ============================================================================
TEAM_CHARACTERISTICS_SCHEMA = {
    "type": "object",
    "required": ["teams"],
    "properties": {
        "year": {"type": "integer", "minimum": 2020, "maximum": 2030},
        "note": {"type": "string"},
        "learning_note": {"type": "string"},
        "teams": {
            "type": "object",
            "patternProperties": {
                ".*": {  # Team names vary in format
                    "type": "object",
                    "required": ["overall_performance"],
                    "properties": {
                        "overall_performance": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "uncertainty": {"type": "number", "minimum": 0, "maximum": 1},
                        "note": {"type": "string"},
                    },
                }
            },
        },
    },
}


# ============================================================================
# TRACK CHARACTERISTICS SCHEMA
# ============================================================================
TRACK_CHARACTERISTICS_SCHEMA = {
    "type": "object",
    "required": ["tracks"],
    "properties": {
        "year": {"type": "integer", "minimum": 2020, "maximum": 2030},
        "note": {"type": "string"},
        "tracks": {
            "type": "object",
            "patternProperties": {
                ".*": {  # Track names vary
                    "type": "object",
                    "required": [],
                    "properties": {
                        "pit_stop_loss": {"type": "number", "minimum": 0},
                        "safety_car_prob": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "overtaking_difficulty": {
                            "type": "number",
                            "minimum": 0,
                            "maximum": 1,
                        },
                        "type": {"type": "string"},
                        "has_sprint": {"type": "boolean"},
                    },
                }
            },
        },
    },
}


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================


def validate_json(data: Dict[str, Any], schema: Dict[str, Any], filename: str) -> None:
    """
    Validate JSON data against schema using jsonschema library.

    Raises ValueError if validation fails.
    """
    if validate is None:
        logger.warning(
            f"jsonschema library not available. Skipping validation of {filename}. "
            "Install jsonschema to enable validation: pip install jsonschema"
        )
        return

    try:
        validate(instance=data, schema=schema)
        logger.info(f"✓ {filename} validated successfully")
    except ValidationError as e:
        # In Python 3, ValidationError.message doesn't exist - use str(e) instead
        error_msg = f"Invalid {filename}: {str(e)}"
        logger.error(f"✗ {filename} validation failed: {error_msg}")
        raise ValueError(error_msg) from e
    except (AttributeError, TypeError, KeyError, ValueError) as e:
        error_msg = f"Unexpected validation error in {filename}: {str(e)}"
        logger.error(f"✗ {error_msg}")
        raise ValueError(error_msg) from e


def validate_driver_characteristics(data: Dict[str, Any]) -> None:
    """
    Validate driver characteristics JSON.
    """
    validate_json(data, DRIVER_CHARACTERISTICS_SCHEMA, "driver_characteristics.json")


def validate_team_characteristics(data: Dict[str, Any]) -> None:
    """
    Validate team characteristics JSON.
    """
    validate_json(data, TEAM_CHARACTERISTICS_SCHEMA, "team_characteristics.json")


def validate_track_characteristics(data: Dict[str, Any]) -> None:
    """
    Validate track characteristics JSON.
    """
    validate_json(data, TRACK_CHARACTERISTICS_SCHEMA, "track_characteristics.json")
