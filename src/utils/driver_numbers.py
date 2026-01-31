"""
Driver Number Mapping for 2026 Season

Maps driver names to their permanent race numbers for the 2026 season.
Updated after lineup announcements December 2025.

Source: F1 official 2026 lineup announcements
Last updated: December 2025

Usage:
    >>> from helpers.driver_number_mapping import get_driver_number
    >>> get_driver_number('Lando Norris')
    4
    >>> get_driver_number('Max Verstappen')
    3
"""

from typing import Optional

# 2026 Driver Numbers
DRIVER_NUMBERS = {
    # McLaren
    "Lando Norris": 4,  # Will use #1 if defending champion
    "Oscar Piastri": 81,
    # Mercedes
    "George Russell": 63,
    "Kimi Antonelli": 12,
    # Red Bull Racing
    "Max Verstappen": 3,  # Will use #1 if defending champion
    "Isack Hadjar": 6,
    # Ferrari
    "Charles Leclerc": 16,
    "Lewis Hamilton": 44,
    # Williams
    "Alex Albon": 23,
    "Carlos Sainz": 55,
    # Racing Bulls
    "Liam Lawson": 30,
    "Arvid Lindblad": 41,
    # Aston Martin
    "Fernando Alonso": 14,
    "Lance Stroll": 18,
    # Haas
    "Esteban Ocon": 31,
    "Oliver Bearman": 87,
    # Audi (formerly Kick Sauber)
    "Nico Hulkenberg": 27,
    "Gabriel Bortoleto": 5,
    # Alpine
    "Pierre Gasly": 10,
    "Franco Colapinto": 43,
    # Cadillac (new team)
    "Valtteri Bottas": 77,
    "Sergio Perez": 11,
}

# Abbreviations (FastF1 format)
DRIVER_ABBREVIATIONS = {
    "NOR": "Lando Norris",
    "PIA": "Oscar Piastri",
    "RUS": "George Russell",
    "ANT": "Kimi Antonelli",
    "VER": "Max Verstappen",
    "HAD": "Isack Hadjar",
    "LEC": "Charles Leclerc",
    "HAM": "Lewis Hamilton",
    "ALB": "Alex Albon",
    "SAI": "Carlos Sainz",
    "LAW": "Liam Lawson",
    "LIN": "Arvid Lindblad",
    "ALO": "Fernando Alonso",
    "STR": "Lance Stroll",
    "OCO": "Esteban Ocon",
    "BEA": "Oliver Bearman",
    "HUL": "Nico Hulkenberg",
    "BOR": "Gabriel Bortoleto",
    "GAS": "Pierre Gasly",
    "COL": "Franco Colapinto",
    "BOT": "Valtteri Bottas",
    "PER": "Sergio Perez",
}


def get_driver_number(driver_name: str, use_champion_number: bool = False) -> Optional[int]:
    """
    Get driver's permanent race number.

    Args:
        driver_name: Full driver name (e.g., 'Lando Norris')
        use_champion_number: If True and driver is defending champion, return 1

    Returns:
        Driver number, or None if unknown

    Example:
        >>> get_driver_number('Lando Norris')
        4
        >>> get_driver_number('Lando Norris', use_champion_number=True)
        1  # If Norris won 2025 championship
    """
    number = DRIVER_NUMBERS.get(driver_name)

    # Special case: defending champion can choose #1
    if use_champion_number and number in [3, 4]:
        # Verstappen (#3) or Norris (#4) - whoever won 2025
        return 1

    return number


def get_driver_from_abbreviation(abbr: str) -> Optional[str]:
    """
    Convert 3-letter abbreviation to full name.

    Args:
        abbr: 3-letter driver code (e.g., 'NOR')

    Returns:
        Full driver name, or None if unknown

    Example:
        >>> get_driver_from_abbreviation('NOR')
        'Lando Norris'
    """
    return DRIVER_ABBREVIATIONS.get(abbr.upper())


def get_all_drivers_2026() -> list[str]:
    """
    Get list of all 2026 drivers.

    Returns:
        List of driver names
    """
    return list(DRIVER_NUMBERS.keys())


def get_team_drivers_2026(team: str) -> list[str]:
    """
    Get driver lineup for a team.

    Args:
        team: Team name (canonical format, e.g., 'MCLAREN')

    Returns:
        List of driver names for that team

    Example:
        >>> get_team_drivers_2026('MCLAREN')
        ['Lando Norris', 'Oscar Piastri']
    """
    lineups = {
        "MCLAREN": ["Lando Norris", "Oscar Piastri"],
        "MERCEDES": ["George Russell", "Kimi Antonelli"],
        "RED BULL": ["Max Verstappen", "Isack Hadjar"],
        "FERRARI": ["Charles Leclerc", "Lewis Hamilton"],
        "WILLIAMS": ["Alex Albon", "Carlos Sainz"],
        "RB": ["Liam Lawson", "Arvid Lindblad"],
        "RACING BULLS": ["Liam Lawson", "Arvid Lindblad"],
        "ASTON MARTIN": ["Fernando Alonso", "Lance Stroll"],
        "HAAS": ["Esteban Ocon", "Oliver Bearman"],
        "AUDI": ["Nico Hulkenberg", "Gabriel Bortoleto"],
        "SAUBER": ["Nico Hulkenberg", "Gabriel Bortoleto"],  # Historical name
        "ALPINE": ["Pierre Gasly", "Franco Colapinto"],
        "CADILLAC": ["Valtteri Bottas", "Sergio Perez"],
    }

    return lineups.get(team.upper(), [])
