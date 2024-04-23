import sys

sys.path.append(".")


from src.aspect_ratio import calculate_aspect_ratio_tag


def test_aspect_ratio():
    assert calculate_aspect_ratio_tag(380, 768) == "<|aspect_ratio:ultra_tall|>"
    assert calculate_aspect_ratio_tag(384, 768) == "<|aspect_ratio:ultra_tall|>"

    assert calculate_aspect_ratio_tag(432, 768) == "<|aspect_ratio:tall|>"
    assert calculate_aspect_ratio_tag(768, 1024) == "<|aspect_ratio:tall|>"

    assert calculate_aspect_ratio_tag(768, 768) == "<|aspect_ratio:square|>"
    assert calculate_aspect_ratio_tag(600, 640) == "<|aspect_ratio:square|>"

    assert calculate_aspect_ratio_tag(1024, 768) == "<|aspect_ratio:wide|>"
    assert calculate_aspect_ratio_tag(1366, 768) == "<|aspect_ratio:wide|>"

    assert calculate_aspect_ratio_tag(1536, 768) == "<|aspect_ratio:ultra_wide|>"
    assert calculate_aspect_ratio_tag(2048, 768) == "<|aspect_ratio:ultra_wide|>"
