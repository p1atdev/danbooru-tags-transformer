import sys

sys.path.append(".")


from src.aspect_ratio import calculate_aspect_ratio_tag


def test_aspect_ratio():
    assert calculate_aspect_ratio_tag(768, 380) == "<|aspect_ratio:ultra_tall|>"
    assert calculate_aspect_ratio_tag(768, 384) == "<|aspect_ratio:ultra_tall|>"

    assert calculate_aspect_ratio_tag(768, 432) == "<|aspect_ratio:tall|>"
    assert calculate_aspect_ratio_tag(1024, 768) == "<|aspect_ratio:tall|>"

    assert calculate_aspect_ratio_tag(768, 768) == "<|aspect_ratio:square|>"
    assert calculate_aspect_ratio_tag(640, 600) == "<|aspect_ratio:square|>"

    assert calculate_aspect_ratio_tag(768, 1024) == "<|aspect_ratio:wide|>"
    assert calculate_aspect_ratio_tag(768, 1366) == "<|aspect_ratio:wide|>"

    assert calculate_aspect_ratio_tag(768, 1536) == "<|aspect_ratio:ultra_wide|>"
    assert calculate_aspect_ratio_tag(768, 2048) == "<|aspect_ratio:ultra_wide|>"
