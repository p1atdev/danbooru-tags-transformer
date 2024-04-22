from .tags import (
    ASPECT_RATIO_SQUARE,
    ASPECT_RATIO_ULTRA_WIDE,
    ASPECT_RATIO_WIDE,
    ASPECT_RATIO_TALL,
    ASPECT_RATIO_ULTRA_TALL,
)


def calculate_aspect_ratio_tag(height: int, width: int):
    """Calculate the aspect ratio tag based on the height and width of the image."""
    aspect_ratio = width / height

    if aspect_ratio <= 1 / 2:
        return ASPECT_RATIO_ULTRA_TALL
    elif aspect_ratio <= 0.9:
        return ASPECT_RATIO_TALL
    elif aspect_ratio < 1 / 0.9:
        return ASPECT_RATIO_SQUARE
    elif aspect_ratio < 2:
        return ASPECT_RATIO_WIDE
    else:
        return ASPECT_RATIO_ULTRA_WIDE
