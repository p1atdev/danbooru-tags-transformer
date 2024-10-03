import math

from .tags import (
    AspectRatioTokens,
)

TOO_TALL = 1 / 5
TOO_WIDE = 5 / 1


def is_extreme_aspect_ratio(width: int, height: int):
    """Check if the aspect ratio is extreme."""

    aspect_ratio = width / height

    return aspect_ratio <= TOO_TALL or aspect_ratio >= TOO_WIDE


def calculate_aspect_ratio_tag(width: int, height: int):
    """Calculate the aspect ratio tag based on the height and width of the image."""
    aspect_ratio = width / height

    if aspect_ratio <= 1 / math.sqrt(3):
        return AspectRatioTokens.ASPECT_RATIO_ULTRA_TALL
    elif aspect_ratio <= 8 / 9:  #
        return AspectRatioTokens.ASPECT_RATIO_TALL
    elif aspect_ratio < 9 / 8:
        return AspectRatioTokens.ASPECT_RATIO_SQUARE
    elif aspect_ratio < math.sqrt(3):
        return AspectRatioTokens.ASPECT_RATIO_WIDE
    else:
        return AspectRatioTokens.ASPECT_RATIO_ULTRA_WIDE
