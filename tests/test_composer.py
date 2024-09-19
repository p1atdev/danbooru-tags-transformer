import sys

sys.path.append(".")

from src.group import TagGroup
from src.cluster import TagCluster
from src.organizer import GroupTagOrganizer
from src.composer import (
    TagComposer,
    PredefinedTags,
    PredefinedTagType,
    TagFrequency,
    TagSorter,
)

KEEP_IDENTITY_TOKEN = "<|keep_identity|>"

group = TagGroup()


def test_composer():
    tag_composer = TagComposer()

    assert tag_composer is not None


def test_load_predifined_tags():
    artistic_error_tags = PredefinedTags.from_txt_file("tags/artistic_error.txt")
    assert artistic_error_tags.tags

    background_tags = PredefinedTags.from_txt_file("tags/background.txt")
    assert background_tags.tags

    ban_meta_tags = PredefinedTags.from_txt_file("tags/ban_meta.txt")
    assert ban_meta_tags.tags

    color_theme_tags = PredefinedTags.from_txt_file("tags/color_theme.txt")
    assert color_theme_tags.tags

    displeasing_meta_tags = PredefinedTags.from_txt_file("tags/displeasing_meta.txt")
    assert displeasing_meta_tags.tags

    focus_tags = PredefinedTags.from_txt_file("tags/focus.txt")
    assert focus_tags.tags

    people_tags = PredefinedTags.from_txt_file("tags/people.txt")
    assert people_tags.tags

    usable_meta_tags = PredefinedTags.from_txt_file("tags/usable_meta.txt")
    assert usable_meta_tags.tags

    watermark_tags = PredefinedTags.from_txt_file("tags/watermark.txt")
    assert watermark_tags.tags


def test_tag_sorter():
    cluster = TagCluster.from_pretrained("data/cluster_map_1440c2.json")
    frequency = TagFrequency.from_json("data/tag_frequency.json")

    sorter = TagSorter(
        cluster,
        frequency,
        high_priority_groups=[
            PredefinedTags.from_txt_file("tags/people.txt"),
            PredefinedTags.from_txt_file("tags/background.txt"),
            PredefinedTags.from_txt_file("tags/watermark.txt"),
            PredefinedTags.from_txt_file("tags/artistic_error.txt"),
        ],
    )

    tags = (
        "watermark, absurdres, simple background, "
        "upper body, shirt, sky, green hair, "
        "short hair, solo, smile, 1girl, "
        "signature, bad anatomy, "
        "iseri nina"
    ).split(", ")
    sorted_tags, excluded, other = sorter.sort_tags(tags)

    print(sorted_tags)
    print(excluded)
    print(other)

    assert sorted_tags == [
        "short hair",
        "green hair",
        "absurdres",
        "shirt",
        "solo",
        "smile",
        "upper body",
        "sky",
    ]
    assert excluded == [
        ["1girl"],
        ["simple background"],
        ["watermark", "signature"],
        ["bad anatomy"],
    ]
    assert other == [
        "iseri nina",
    ]
