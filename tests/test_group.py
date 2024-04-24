import sys

sys.path.append(".")

from src.group import TagGroup


def test_tag_group():
    group = TagGroup()

    assert group.path is not None


def test_people_tags():
    group = TagGroup()

    assert len(group.people_tags) > 0
    assert "1girl" in group.people_tags
    assert "6+girls" in group.people_tags


def test_watermark_tags():
    group = TagGroup()

    assert len(group.watermark_tags) > 0
    assert "watermark" in group.watermark_tags


def test_artistic_error_tags():
    group = TagGroup()

    assert len(group.artistic_error_tags) > 0
    assert "bad anatomy" in group.artistic_error_tags


def test_focus_tags():
    group = TagGroup()

    assert len(group.focus_tags) > 0
    assert "solo focus" in group.focus_tags
