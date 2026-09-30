"""Acquirer profiles, selected by the universe CSV's Instrument ID.

Profile text uses the company's item_text.
"""

import csv

from .config import UNIVERSE_FILE
from .items import item_text

DEFAULT_PROFILE = "DIS"


def _load_acquirer(name: str) -> dict:
    """The acquirer's row of the universe CSV; ValueError if it is not there."""
    with open(UNIVERSE_FILE, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("Instrument") == name:
                return row
    raise ValueError(f"Unknown acquirer {name!r}: not in {UNIVERSE_FILE}")


def get_acquirer_name(name: str) -> str:
    return _load_acquirer(name).get("CompanyName", "Unknown")


def get_profile_text(name: str) -> str:
    return item_text(_load_acquirer(name))


def profile_tag(name: str) -> str:
    """The profile's token in archive filenames: the item ID with dots removed."""
    return name.replace(".", "")
