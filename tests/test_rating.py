import sys

sys.path.append(".")

from src.rating import get_rating_tag, get_ambitious_rating_tag


def test_get_rating_tag():
    assert get_rating_tag("g") == "<|rating:general|>"
    assert get_rating_tag("s") == "<|rating:sensitive|>"
    assert get_rating_tag("q") == "<|rating:questionable|>"
    assert get_rating_tag("e") == "<|rating:explicit|>"


def test_get_ambitious_rating_tag():
    assert get_ambitious_rating_tag("g") == "<|rating:sfw|>"
    assert get_ambitious_rating_tag("s") == "<|rating:sfw|>"
    assert get_ambitious_rating_tag("q") == "<|rating:nsfw|>"
    assert get_ambitious_rating_tag("e") == "<|rating:nsfw|>"
