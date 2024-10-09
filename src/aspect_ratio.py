import math

from .tags import (
    AspectRatioTokens,
)

TOO_TALL = -1.25
TOO_WIDE = 1.25


def is_extreme_aspect_ratio(width: int, height: int):
    """Check if the aspect ratio is extreme."""

    aspect_ratio = math.log2(width / height)

    return aspect_ratio <= TOO_TALL or aspect_ratio >= TOO_WIDE


def calculate_aspect_ratio_tag(width: int, height: int):
    """Calculate the aspect ratio tag based on the height and width of the image."""
    aspect_ratio = math.log2(width / height)  # log 2 of the aspect ratio

    if aspect_ratio <= -1.25:
        return AspectRatioTokens.ASPECT_RATIO_TOO_TALL
    elif aspect_ratio <= -0.75:
        return AspectRatioTokens.ASPECT_RATIO_TALL_WALLPAPER
    elif aspect_ratio <= -0.25:
        return AspectRatioTokens.ASPECT_RATIO_TALL
    elif aspect_ratio < 0.25:
        return AspectRatioTokens.ASPECT_RATIO_SQUARE
    elif aspect_ratio < 0.75:
        return AspectRatioTokens.ASPECT_RATIO_WIDE
    elif aspect_ratio < 1.25:
        return AspectRatioTokens.ASPECT_RATIO_WIDE_WALLPAPER
    else:
        return AspectRatioTokens.ASPECT_RATIO_TOO_WIDE
