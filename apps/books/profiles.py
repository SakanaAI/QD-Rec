"""
Query-user profiles for the Goodreads-Children app.

The two query users (config.USERS) are the paper's two readers. Each profile text is
`data/<key>/profile.txt`, which `data_prep.py` builds from the reader's own reviews.
"""

from __future__ import annotations

from .config import USERS, profile_file, require_user

DEFAULT_PROFILE = "picture_books_reader"


def list_user_keys() -> list[str]:
    return list(USERS)


def get_user_id(key: str) -> str:
    return require_user(key)["id"]


def get_user_name(key: str) -> str:
    return require_user(key)["name"]


def get_profile_text(key: str) -> str:
    p = profile_file(key)
    if not p.exists():
        raise FileNotFoundError(
            f"Profile not built for {key!r}: {p}. Run `data_prep` first."
        )
    return p.read_text(encoding="utf-8")


def profile_tag(key: str) -> str:
    """The profile's token in archive filenames: the reader key itself."""
    return key
