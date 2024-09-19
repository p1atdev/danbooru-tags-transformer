from .tags import (
    AspectRatioTokens,
)


def calculate_aspect_ratio_tag(width: int, height: int):
    """Calculate the aspect ratio tag based on the height and width of the image."""
    aspect_ratio = width / height

    if aspect_ratio < 1 / 4:
        return AspectRatioTokens.ASPECT_RATIO_ULTRA_TALL
    elif aspect_ratio < 8 / 9:  #
        return AspectRatioTokens.ASPECT_RATIO_TALL
    elif aspect_ratio < 9 / 8:
        return AspectRatioTokens.ASPECT_RATIO_SQUARE
    elif aspect_ratio < 4:
        return AspectRatioTokens.ASPECT_RATIO_WIDE
    else:
        return AspectRatioTokens.ASPECT_RATIO_ULTRA_WIDE
