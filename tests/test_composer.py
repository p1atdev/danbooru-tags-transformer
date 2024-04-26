import sys

sys.path.append(".")

from src.group import TagGroup
from src.cluster import TagCluster
from src.organizer import GroupTagOrganizer
from src.composer import TagComposer

KEEP_IDENTITY_TOKEN = "<|keep_identity|>"

group = TagGroup()


def test_composer():
    tag_composer = TagComposer()

    assert tag_composer is not None


def test_composer_recompose_tags():
    organizer = GroupTagOrganizer(group)

    tag_composer = TagComposer()

    result = organizer.organize_tags(
        [
            "1girl",  # people tag
            "solo",  # focus tag
            "solo focus",  # focus tag
            "blonde hair",  # other tag
            "bad anatomy",  # artistic error tag
            "watermark",  # watermark tag
        ]
    )

    pre_tags, post_tags = tag_composer.recompose_tags(result)

    assert "1girl" in pre_tags
    assert "watermark" not in pre_tags and "watermark" not in post_tags


def test_composer_get_common_tags():
    tag_composer = TagComposer(
        fuzzy_rating_tag_rate=0,
    )

    common = tag_composer.get_common_tags(
        rating="g",
        general=["1girl", "cat ears"],
        image_width=896,
        image_height=1152,
    )

    assert common.length_tag == "<|length:very_short|>"
    assert common.rating_tag == "<|rating:general|>"
    assert common.aspect_ratio_tag == "<|aspect_ratio:tall|>"


def test_composer_get_common_tags_with_fuzzy_rating_tag():
    tag_composer = TagComposer(fuzzy_rating_tag_rate=1)

    common = tag_composer.get_common_tags(
        rating="g",
        general=["1girl", "cat ears"],
        image_width=896,
        image_height=1152,
    )

    assert common.rating_tag == "<|rating:sfw|>"


def test_composer_get_keep_identity_condition_part():
    organizer = GroupTagOrganizer(group)

    tag_composer = TagComposer(fuzzy_rating_tag_rate=0)

    result = organizer.organize_tags(["1girl", "cat ears"])

    common = tag_composer.get_keep_identity_condition_part(
        copyright=["vocaloid"],
        character=["hatsune miku"],
        organizer_result=result,
    )

    assert common.copyright_part == ["vocaloid"]
    assert common.character_part == ["hatsune miku"]
    assert "1girl" in common.pre_general_part


def test_composer_get_free_condition_part():
    organizer = GroupTagOrganizer(group)

    tag_composer = TagComposer(
        fuzzy_rating_tag_rate=0,
        drop_people_rate=0,
    )

    result = organizer.organize_tags(["1girl", "cat ears", "blue hair"])

    common = tag_composer.get_free_condition_part(
        copyright=["vocaloid"],
        character=["hatsune miku"],
        organizer_result=result,
    )

    assert common.copyright_part == ["vocaloid"]
    assert common.character_part == ["hatsune miku"]
    assert "1girl" in common.pre_general_part
    assert common.post_general_part == sorted(common.post_general_part)


def test_composer_get_components_identity_keep():
    organizer = GroupTagOrganizer(group)

    tag_composer = TagComposer(
        fuzzy_rating_tag_rate=0,
        drop_people_rate=0,
    )

    result = organizer.organize_tags(["1girl", "cat ears", "blue hair"])

    components = tag_composer.get_components_identity_keep(
        rating="g",
        copyright=["aaa"],
        character=["bbb"],
        organizer_result=result,
        image_width=896,
        image_height=1152,
    )

    assert components.copyright == "aaa"
    assert components.character == "bbb"
    assert components.rating == "<|rating:general|>"
    assert components.aspect_ratio == "<|aspect_ratio:tall|>"
    assert components.length == "<|length:very_short|>"
    assert "1girl" in components.general_condition


def test_composer_get_components_identity_free():
    organizer = GroupTagOrganizer(group)

    tag_composer = TagComposer(
        fuzzy_rating_tag_rate=0,
        drop_people_rate=0,
    )

    result = organizer.organize_tags(["1girl", "cat ears", "blue hair"])

    components = tag_composer.get_components_identity_free(
        rating="g",
        copyright=["aaa"],
        character=["bbb"],
        organizer_result=result,
        image_width=896,
        image_height=1152,
    )

    assert components.copyright == "aaa"
    assert components.character == "bbb"
    assert components.rating == "<|rating:general|>"
    assert components.aspect_ratio == "<|aspect_ratio:tall|>"
    assert components.length == "<|length:very_short|>"
    assert "1girl" in components.general_condition
