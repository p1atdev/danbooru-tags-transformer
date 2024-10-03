import sys

sys.path.append(".")

from src.formatter import format_completion
from src.tags import LengthTokens, AspectRatioTokens, RatingTokens


def test_format_pretrain():
    prompt = format_completion(
        priority="1girl, solo, simple background",
        general="my general tag",
        copyright="my copyright tag",
        character="my original character",
        rating=RatingTokens.RATING_GENERAL,
        aspect_ratio=AspectRatioTokens.ASPECT_RATIO_TALL,
        length=LengthTokens.LENGTH_MEDIUM,
        meta="my meta tag",
    )

    assert prompt == (
        "<|bos|>"
        "<copyright>my copyright tag</copyright>"
        "<character>my original character</character>"
        "<|rating:general|><|aspect_ratio:tall|><|length:medium|>"
        "<general>1girl, solo, simple background, my meta tag, my general tag</general>"
        "<|eos|>"
    )
