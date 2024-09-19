from typing import Literal, Union

from .tags import (
    RatingTokens,
)

SHORT_RATING_TAG = Literal["g", "s", "q", "e"]

LONG_RATING_TAG_MAP: dict[SHORT_RATING_TAG, str] = {
    "g": RatingTokens.RATING_GENERAL,
    "s": RatingTokens.RATING_SENSITIVE,
    "q": RatingTokens.RATING_QUESTIONABLE,
    "e": RatingTokens.RATING_EXPLICIT,
}

AMBITIOUS_RATING_TAG_MAP: dict[SHORT_RATING_TAG, str] = {
    "g": RatingTokens.RATING_SFW,
    "s": RatingTokens.RATING_SFW,
    "q": RatingTokens.RATING_NSFW,
    "e": RatingTokens.RATING_NSFW,
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
