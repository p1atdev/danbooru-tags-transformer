from .tags import (
    LengthTokens,
)


def get_length_tag(length: int):
    """Get the length tag from the length."""

    if length < 7:
        return LengthTokens.LENGTH_VERY_SHORT
    elif length < 25:
        return LengthTokens.LENGTH_SHORT
    elif length < 32:
        return LengthTokens.LENGTH_MEDIUM
    elif length < 48:
        return LengthTokens.LENGTH_LONG
    else:
        return LengthTokens.LENGTH_VERY_LONG
