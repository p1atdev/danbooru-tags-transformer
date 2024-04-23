import sys

sys.path.append(".")

from src.group import TagGroup
from src.cluster import TagCluster
from src.organizer import TagOrganizer
from src.composer import TagComposer

KEEP_IDENTITY_TOKEN = "<|keep_identity|>"

group = TagGroup()
cluster = TagCluster.from_pretrained("data/cluster_map.json")


def test_composer():
    organizer = TagOrganizer(group, cluster)

    tag_composer = TagComposer(
        organizer,
        keep_identity_token=KEEP_IDENTITY_TOKEN,
    )

    assert tag_composer.keep_identity_token == KEEP_IDENTITY_TOKEN


def test_composer_recompose_tags():
    organizer = TagOrganizer(group, cluster)

    tag_composer = TagComposer(
        organizer,
        keep_identity_token=KEEP_IDENTITY_TOKEN,
    )

    pre_tags, post_tags = tag_composer.recompose_tags(
        tags=[
            "1girl",  # people tag
            "solo",  # focus tag
            "solo focus",  # focus tag
            "blonde hair",  # other tag
            "bad anatomy",  # artistic error tag
            "watermark",  # watermark tag
        ]
    )

    assert "1girl" in pre_tags
    assert "watermark" not in pre_tags and "watermark" not in post_tags


def test_composer_get_common_tags():
    organizer = TagOrganizer(group, cluster)

    tag_composer = TagComposer(
        organizer,
        keep_identity_token=KEEP_IDENTITY_TOKEN,
        fuzzy_rating_rate=0,
    )

    common = tag_composer.get_common_tags(
        rating="g",
        general=["1girl", "cat ears"],
        image_width=896,
        image_height=1152,
    )

    assert common.length_tag == "<|legnth:very_short|>"
    assert common.rating_tag == "<|rating:general|>"
    assert common.aspect_ratio_tag == "<|aspect_ratio:tall|>"


def test_composer_get_keep_identity_condition_part():
    organizer = TagOrganizer(group, cluster)

    tag_composer = TagComposer(
        organizer,
        keep_identity_token=KEEP_IDENTITY_TOKEN,
        fuzzy_rating_rate=0,
    )

    common = tag_composer.get_keep_identity_condition_part(
        copyright=["vocaloid"],
        character=["hatsune miku"],
        general=["1girl", "cat ears"],
    )

    assert common.copyright_part == ["vocaloid"]
    assert common.character_part == ["hatsune miku"]
    assert "1girl" in common.pre_general_part


def test_composer_get_free_condition_part():
    organizer = TagOrganizer(group, cluster)

    tag_composer = TagComposer(
        organizer,
        keep_identity_token=KEEP_IDENTITY_TOKEN,
        fuzzy_rating_rate=0,
        drop_people_rate=0,
    )

    common = tag_composer.get_free_condition_part(
        copyright=["vocaloid"],
        character=["hatsune miku"],
        general=["1girl", "cat ears", "blue hair"],
    )

    assert common.copyright_part == ["vocaloid"]
    assert common.character_part == ["hatsune miku"]
    assert "1girl" in common.pre_general_part
    assert common.post_general_part == sorted(common.post_general_part)


def test_composer_compose_sft_prompt():
    organizer = TagOrganizer(group, cluster)

    tag_composer = TagComposer(
        organizer,
        keep_identity_token=KEEP_IDENTITY_TOKEN,
        fuzzy_rating_rate=0,
        drop_people_rate=0,
    )

    prompt = tag_composer.compose_sft_prompt(
        rating="g",
        copyright=["aaa"],
        character=["bbb"],
        general=["1girl", "cat ears", "blue hair"],
        image_width=896,
        image_height=1152,
    )

    assert prompt.startswith(
        "<copyright>aaa</copyright><character>bbb</character><|rating:general|><|aspect_ratio:tall|><|legnth:very_short|><general>"
    )


def test_composer_compose_prompt():
    organizer = TagOrganizer(group, cluster)

    tag_composer = TagComposer(
        organizer,
        keep_identity_token=KEEP_IDENTITY_TOKEN,
        fuzzy_rating_rate=0,
        drop_people_rate=0,
    )

    prompt = tag_composer.compose_prompt(
        rating="g",
        copyright=["aaa"],
        character=["bbb"],
        general=["1girl", "cat ears", "blue hair"],
        image_width=896,
        image_height=1152,
    )

    assert (
        prompt
        == "<copyright>aaa</copyright><character>bbb</character><|rating:general|><|aspect_ratio:tall|><|legnth:very_short|><general>1girl, blue hair, cat ears</general>"
    )
