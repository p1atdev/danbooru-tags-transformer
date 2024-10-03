from .tags import (
    LengthTokens,
)

TOO_LONG_CHARACTERS = 10
TOO_LONG_GENERAL = 128


def is_too_long_tags(character_tags: list[str], general_tags: list[str]):
    """Check if the character and general tags are too long."""

    return (
        len(character_tags) > TOO_LONG_CHARACTERS
        or len(general_tags) > TOO_LONG_GENERAL
    )


def get_length_tag(length: int):
    """Get the length tag from the length."""

    if length <= 16:  # 15% (+15%)
        return LengthTokens.LENGTH_VERY_SHORT
    elif length <= 22:  # 35% (+20%)
        return LengthTokens.LENGTH_SHORT
    elif length <= 33:  # 65% (+30%)
        return LengthTokens.LENGTH_MEDIUM
    elif length <= 44:  # 85% (+20%)
        return LengthTokens.LENGTH_LONG
    else:  # 100% (+15%)
        return LengthTokens.LENGTH_VERY_LONG
