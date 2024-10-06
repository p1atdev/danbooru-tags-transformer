from .tags import (
    CommonSpecialTokens,
    TagCategoryTokens,
    InstructionTokens,
    MultiModalTokens,
    RatingTokens,
    AspectRatioTokens,
    LengthTokens,
)

BOS = CommonSpecialTokens.BOS_TOKEN
EOS = CommonSpecialTokens.EOS_TOKEN
GENERAL_START = TagCategoryTokens.GENERAL_START
GENERAL_END = TagCategoryTokens.GENERAL_END
COPYRIGHT_START = TagCategoryTokens.COPYRIGHT_START
COPYRIGHT_END = TagCategoryTokens.COPYRIGHT_END
CHARACTER_START = TagCategoryTokens.CHARACTER_START
CHARACTER_END = TagCategoryTokens.CHARACTER_END
RATING_START = TagCategoryTokens.RATING_START
RATING_END = TagCategoryTokens.RATING_END

# INPUT_START = InstructionTokens.INPUT_START
INPUT_END = InstructionTokens.INPUT_END
USE_START = InstructionTokens.USE_START
USE_END = InstructionTokens.USE_END
BAN_START = InstructionTokens.BAN_START
BAN_END = InstructionTokens.BAN_END
PROJECTION_START = MultiModalTokens.PROJECTION_START
# PROJECTION_END = MultiModalTokens.PROJECTION_END
LINEART_START = MultiModalTokens.LINEART_START

# 後続の生成をするテンプレート
TAG_COMPLETION_TEMPLATE = (
    f"{BOS}"
    #
    "{rating}"
    "{aspect_ratio}"
    "{length}"
    #
    f"{COPYRIGHT_START}"
    "{copyright}"
    f"{COPYRIGHT_END}"
    #
    f"{CHARACTER_START}"
    "{character}"
    f"{CHARACTER_END}"
    #
    f"{GENERAL_START}"
    "{priority_meta_general}"
    f"{GENERAL_END}"
    #
    f"{EOS}"
).strip()

# 初期条件指示を含むテンプレート
TAG_INITIAL_CONDITION_TEMPLATE = (
    f"{BOS}"
    #
    "{rating_aspect_ratio_length}"  # shuffle
    #
    f"{COPYRIGHT_START}"
    "{copyright}"
    f"{COPYRIGHT_END}"
    #
    f"{CHARACTER_START}"
    "{character}"
    f"{CHARACTER_END}"
    #
    f"{GENERAL_START}"
    "{condition}"
    f"{INPUT_END}"  #! instruction end
    "{meta_general}"
    f"{GENERAL_END}"
    #
    f"{EOS}"
).strip()

TAG_RESTRICTION_CONDITION_TEMPLATE = (
    f"{BOS}"
    #
    "{rating_aspect_ratio_length}"  # shuffle
    #
    f"{COPYRIGHT_START}"
    "{copyright}"
    f"{COPYRIGHT_END}"
    #
    f"{CHARACTER_START}"
    "{character}"
    f"{CHARACTER_END}"
    # restriction
    f"{USE_START}"
    "{use}"
    f"{USE_END}"
    f"{BAN_START}"
    "{ban}"
    f"{BAN_END}"
    #
    f"{GENERAL_START}"
    f"{INPUT_END}"  #! instruction end
    "{priority}, {meta}, {general}"
    f"{GENERAL_END}"
    #
    f"{EOS}"
).strip()

TAG_PROJECTION_CONDITION_TEMPLATE = (
    f"{BOS}"
    # projection
    f"{PROJECTION_START}"
    #
    "{rating_aspect_ratio_length}"  # shuffle
    #
    f"{INPUT_END}"  #! instruction end
    #
    f"{COPYRIGHT_START}"
    "{copyright}"
    f"{COPYRIGHT_END}"
    #
    f"{CHARACTER_START}"
    "{character}"
    f"{CHARACTER_END}"
    #
    f"{GENERAL_START}"
    "{priority}, {meta}, {general}"
    f"{GENERAL_END}"
    #
    f"{EOS}"
).strip()

TAG_LINEART_CONDITION_TEMPLATE = (
    f"{BOS}"
    # projection
    f"{LINEART_START}"
    #
    "{rating_aspect_ratio_length}"  # shuffle
    #
    f"{INPUT_END}"  #! instruction end
    #
    f"{COPYRIGHT_START}"
    "{copyright}"
    f"{COPYRIGHT_END}"
    #
    f"{CHARACTER_START}"
    "{character}"
    f"{CHARACTER_END}"
    #
    f"{GENERAL_START}"
    "{priority}, {meta}, {general}"
    f"{GENERAL_END}"
    #
    f"{EOS}"
).strip()


def format_completion(
    priority: list[str],  # high priority tags
    general: list[str],
    copyright: list[str],
    character: list[str],
    meta: list[str],
    rating: str,
    aspect_ratio: str,
    length: str,
):
    return TAG_COMPLETION_TEMPLATE.format(
        priority_meta_general=", ".join(priority + meta + general),
        copyright=", ".join(copyright),
        character=", ".join(character),
        rating=rating,
        aspect_ratio=aspect_ratio,
        length=length,
    )


def format_sft_with_initial_condition(
    condition: list[str],
    copyright: list[str],
    character: list[str],
    meta_general: list[str],
    rating_aspect_ratio_length: list[str],
):
    return TAG_INITIAL_CONDITION_TEMPLATE.format(
        rating_aspect_ratio_length="".join(rating_aspect_ratio_length),
        condition=", ".join(condition),
        copyright=", ".join(copyright),
        character=", ".join(character),
        meta_general=", ".join(meta_general),
    )
