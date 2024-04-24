import sys

sys.path.append(".")

from src.composer import PrompotComponents
from src.formatter import format_pretrain, format_sft


def test_format_pretrain():
    components = PrompotComponents(
        copyright="my copyright tag",
        character="my original character",
        rating="<|rating:general|>",
        aspect_ratio="<|aspect_ratio:tall|>",
        length="<|length:very_short|>",
        general_condition="this field should not be included",
        general_completion="1girl, animal ears, blue hair, cat ears",
    )

    prompt = format_pretrain(components)

    assert prompt == (
        "<|bos|>"
        "<copyright>my copyright tag</copyright>"
        "<character>my original character</character>"
        "<|rating:general|><|aspect_ratio:tall|><|length:very_short|>"
        "<general>1girl, animal ears, blue hair, cat ears</general>"
        "<|eos|>"
    )


def test_format_sft():
    components = PrompotComponents(
        copyright="vocaloid",
        character="hatsune miku",
        rating="<|rating:general|>",
        aspect_ratio="<|aspect_ratio:wide|>",
        length="<|length:very_short|>",
        general_condition="solo, 1girl",
        general_completion="blue hair, long hair, looking at viewer, twintails",
    )

    prompt = format_sft(components, "<|identity_level:lax|>")

    assert prompt == (
        "<|bos|>"
        "<copyright>vocaloid</copyright>"
        "<character>hatsune miku</character>"
        "<|rating:general|><|aspect_ratio:wide|><|length:very_short|>"
        "<general>solo, 1girl<|identity_level:lax|><|input_end|>blue hair, long hair, looking at viewer, twintails</general>"
        "<|eos|>"
    )
