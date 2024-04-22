from typing import Literal, Union

from .tags import (
    RATING_GENERAL,
    RATING_SENSITIVE,
    RATING_QUESTIONABLE,
    RATING_EXPLICIT,
    RATING_SFW,
    RATING_NSFW,
)

SHORT_RATING_TAG = Literal["g", "s", "q", "e"]

LONG_RATING_TAG_MAP: dict[SHORT_RATING_TAG, str] = {
    "g": RATING_GENERAL,
    "s": RATING_SENSITIVE,
    "q": RATING_QUESTIONABLE,
    "e": RATING_EXPLICIT,
}

AMBITIOUS_RATING_TAG_MAP: dict[SHORT_RATING_TAG, str] = {
    "g": RATING_SFW,
    "s": RATING_SFW,
    "q": RATING_NSFW,
    "e": RATING_NSFW,
}


def get_rating_tag(rating: SHORT_RATING_TAG):
    """Get the long rating tag from the short rating tag."""
    if rating not in LONG_RATING_TAG_MAP:
        raise ValueError(f"Invalid rating tag: {rating}")
    return LONG_RATING_TAG_MAP[rating]


def get_ambitious_rating_tag(rating: SHORT_RATING_TAG):
    """Get the ambitious rating tag from the short rating tag."""
    if rating not in AMBITIOUS_RATING_TAG_MAP:
        raise ValueError(f"Invalid rating tag: {rating}")
    return AMBITIOUS_RATING_TAG_MAP[rating]
