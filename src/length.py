from .tags import (
    LENGTH_VERY_SHORT,
    LENGTH_SHORT,
    LENGTH_MEDIUM,
    LENGTH_LONG,
    LENGTH_VERY_LONG,
)


def get_length_tag(length: int):
    """Get the length tag from the length."""

    if length < 10:
        return LENGTH_VERY_SHORT
    elif length < 20:
        return LENGTH_SHORT
    elif length < 30:
        return LENGTH_MEDIUM
    elif length < 45:
        return LENGTH_LONG
    else:
        return LENGTH_VERY_LONG
