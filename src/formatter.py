from .composer import PrompotComponents
from .tags import (
    CommonSpecialTokens,
    TagCategoryTokens,
    InstructionTokens,
)


def format_pretrain(components: PrompotComponents) -> str:
    """Format prompt for pretraining."""
    prompt = (
        f"{CommonSpecialTokens.BOS_TOKEN}"
        f"{TagCategoryTokens.COPYRIGHT_START}{components.copyright}{TagCategoryTokens.COPYRIGHT_END}"
        f"{TagCategoryTokens.CHARACTER_START}{components.character}{TagCategoryTokens.CHARACTER_END}"
        f"{components.rating}{components.aspect_ratio}{components.length}"
        f"{TagCategoryTokens.GENERAL_START}{components.general_completion}{TagCategoryTokens.GENERAL_END}"  # no condition (input) part
        f"{CommonSpecialTokens.EOS_TOKEN}"
    )
    return prompt


def format_sft(components: PrompotComponents, identity_level_tag: str) -> str:
    """Format prompt for SFT."""
    prompt = (
        f"{CommonSpecialTokens.BOS_TOKEN}"
        f"{TagCategoryTokens.COPYRIGHT_START}{components.copyright}{TagCategoryTokens.COPYRIGHT_END}"
        f"{TagCategoryTokens.CHARACTER_START}{components.character}{TagCategoryTokens.CHARACTER_END}"
        f"{components.rating}{components.aspect_ratio}{components.length}"
        f"{TagCategoryTokens.GENERAL_START}{components.general_condition}"
        f"{identity_level_tag}{InstructionTokens.INPUT_END}"
        f"{components.general_completion}{TagCategoryTokens.GENERAL_END}"
        f"{CommonSpecialTokens.EOS_TOKEN}"
    )
    return prompt
