from .composer import PrompotComponents
from .tags import (
    BOS_TOKEN,
    EOS_TOKEN,
    COPYRIGHT_START,
    COPYRIGHT_END,
    CHARACTER_START,
    CHARACTER_END,
    GENERAL_START,
    GENERAL_END,
    INPUT_END,
)


def format_pretrain(components: PrompotComponents) -> str:
    """Format prompt for pretraining."""
    prompt = (
        f"{BOS_TOKEN}"
        f"{COPYRIGHT_START}{components.copyright}{COPYRIGHT_END}"
        f"{CHARACTER_START}{components.character}{CHARACTER_END}"
        f"{components.rating}{components.aspect_ratio}{components.length}"
        f"{GENERAL_START}{components.general_completion}{GENERAL_END}"  # no condition (input) part
        f"{EOS_TOKEN}"
    )
    return prompt


def format_sft(components: PrompotComponents, identity_level_tag: str) -> str:
    """Format prompt for SFT."""
    prompt = (
        f"{BOS_TOKEN}"
        f"{COPYRIGHT_START}{components.copyright}{COPYRIGHT_END}"
        f"{CHARACTER_START}{components.character}{CHARACTER_END}"
        f"{components.rating}{components.aspect_ratio}{components.length}"
        f"{GENERAL_START}{components.general_condition}"
        f"{identity_level_tag}{INPUT_END}"
        f"{components.general_completion}{GENERAL_END}"
        f"{EOS_TOKEN}"
    )
    return prompt
