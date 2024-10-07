import sys

sys.path.append(".")

from src.cluster import TagCluster
from src.composer import (
    TagComposer,
    PredefinedTags,
    PredefinedTagType,
    TagFrequency,
    TagSelector,
)

KEEP_IDENTITY_TOKEN = "<|keep_identity|>"


def test_load_predifined_tags():
    artistic_error_tags = PredefinedTags.from_txt_file(
        "tags/artistic_error.txt", PredefinedTagType.REMOVE
    )
    assert artistic_error_tags.tags

    background_tags = PredefinedTags.from_txt_file(
        "tags/background.txt", PredefinedTagType.REMOVE
    )
    assert background_tags.tags

    ban_meta_tags = PredefinedTags.from_txt_file(
        "tags/ban_meta.txt", PredefinedTagType.REMOVE
    )
    assert ban_meta_tags.tags

    color_theme_tags = PredefinedTags.from_txt_file(
        "tags/color_theme.txt", PredefinedTagType.REMOVE
    )
    assert color_theme_tags.tags

    displeasing_meta_tags = PredefinedTags.from_txt_file(
        "tags/displeasing_meta.txt", PredefinedTagType.REMOVE
    )
    assert displeasing_meta_tags.tags

    focus_tags = PredefinedTags.from_txt_file(
        "tags/focus.txt", PredefinedTagType.REMOVE
    )
    assert focus_tags.tags

    people_tags = PredefinedTags.from_txt_file(
        "tags/people.txt", PredefinedTagType.REMOVE
    )
    assert people_tags.tags

    usable_meta_tags = PredefinedTags.from_txt_file(
        "tags/usable_meta.txt", PredefinedTagType.REMOVE
    )
    assert usable_meta_tags.tags

    watermark_tags = PredefinedTags.from_txt_file(
        "tags/watermark.txt", PredefinedTagType.REMOVE
    )
    assert watermark_tags.tags


def test_people_tags():
    tags = PredefinedTags.people().tags

    assert len(tags) > 0
    assert "1girl" in tags
    assert "6+girls" in tags


def test_watermark_tags():
    group = PredefinedTags.watermark().tags

    assert len(group) > 0
    assert "watermark" in group
    assert "weibo watermark" in group
    assert "sample watermark" in group
    assert "too many watermarks" in group


def test_artistic_error_tags():
    group = PredefinedTags.artistic_error().tags

    assert len(group) > 0
    assert "bad anatomy" in group


def test_focus_tags():
    group = PredefinedTags.focus().tags

    assert len(group) > 0
    assert "solo focus" in group


def test_tag_clustering():
    cluster = TagCluster.from_pretrained("data/cluster_map_1024c1.json")
    frequency = TagFrequency.from_json("data/tag_frequency.json")

    sorter = TagSelector(
        cluster,
        frequency,
        high_priority_groups=[],
    )

    clusters = sorter.clustering_tags(
        [
            "1girl",
            "upper body",
            "staff",
            "holding staff",
            ":d",
            "sitting",
            "medium hair",
        ]
    )

    print(clusters)
    assert clusters == {
        189: ["1girl", ":d"],
        467: ["upper body"],
        45: ["staff", "holding staff"],
        391: ["sitting"],
        992: ["medium hair"],
    }


def test_tag_selector():
    cluster = TagCluster.from_pretrained("data/cluster_map_1024c1.json")
    frequency = TagFrequency.from_json("data/tag_frequency.json")

    sorter = TagSelector(
        cluster,
        frequency,
        high_priority_groups=[
            PredefinedTags.from_txt_file(
                "tags/people.txt", PredefinedTagType.INSERT_START
            ),
            PredefinedTags.from_txt_file(
                "tags/background.txt", PredefinedTagType.INSERT_START
            ),
            PredefinedTags.from_txt_file(
                "tags/watermark.txt", PredefinedTagType.INSERT_START
            ),
            PredefinedTags.from_txt_file(
                "tags/artistic_error.txt", PredefinedTagType.INSERT_START
            ),
        ],
    )

    tags = (
        "watermark, simple background, "
        "upper body, shirt, sky, green hair, "
        "short hair, solo, smile, 1girl, blue eyes, "
        "signature, bad anatomy, breasts, animal ears"
    ).split(", ")
    priorities, conditioning, sorted_tags = sorter.separate_and_sort_tags(
        tags, condition_rate=0.0, temperature=1.0
    )

    print(priorities)
    print(conditioning)
    print(sorted_tags)

    assert priorities == [
        ["1girl"],
        ["simple background"],
        ["watermark", "signature"],
        ["bad anatomy"],
    ]
    assert conditioning == []
    assert sorted_tags == [
        "short hair",
        "solo",
        "blue eyes",
        "breasts",
        "smile",
        "shirt",
        "green hair",
        "animal ears",
        "upper body",
        "sky",
    ]


def test_tag_meta_remove():
    predefined_meta_tags: list[PredefinedTags] = [
        PredefinedTags.ban_meta(),
        PredefinedTags.displeasing_meta(),
        PredefinedTags.usable_meta(),
        PredefinedTags.medium(),
    ]

    meta_tags = [
        "watercolor (medium)",
        "character request",
        "highres",
        "duplicate",
    ]

    banned = []
    removed = []
    inserted = []

    for predefined in predefined_meta_tags:
        for tag_part in predefined.tags:
            for tag in meta_tags:  # 部分的にでも含まれていたら
                if tag_part in tag:
                    if predefined.tag_type == PredefinedTagType.BAN:
                        # BAN row
                        banned.append(tag)
                    elif predefined.tag_type == PredefinedTagType.REMOVE:
                        # just remove
                        removed.append(tag)
                    elif predefined.tag_type == PredefinedTagType.INSERT_START:
                        # do nothing
                        inserted.append(tag)

    assert banned == ["duplicate"]
    assert removed == ["character request"]
    assert inserted == [
        "highres",
        "watercolor (medium)",
    ]


def test_prompt_compose_pretrain():
    cluster = TagCluster.from_pretrained("data/cluster_map_1024c1.json")
    frequency = TagFrequency.from_json("data/tag_frequency.json")

    composer = TagComposer(
        cluster,
        frequency,
    )

    prompt = composer.compose_pretrain_list(
        general_tags="full body, sitting, solo, watermark, 1girl, weibo watermark".split(
            ", "
        ),  # type: ignore
        copyright_tags="vocaloid".split(", "),  # type: ignore
        character_tags="hatsune miku".split(", "),  # type: ignore
        meta_tags="watercolor (medium), character request, highres".split(", "),  # type: ignore
        rating="g",
        image_width=832,
        image_height=1152,
        temperature=1.0,
        condition_rate=0.0,
    )

    assert prompt == (
        "<|bos|>"
        "<|rating:general|><|aspect_ratio:tall|><|length:very_short|>"
        "<copyright>vocaloid</copyright>"
        "<character>hatsune miku</character>"
        "<general>1girl, solo, watercolor (medium), sitting, full body</general>"
        "<|eos|>"
    )


def test_prompt_compose_pretrain_list():
    cluster = TagCluster.from_pretrained("data/cluster_map_1024c1.json")
    frequency = TagFrequency.from_json("data/tag_frequency.json")

    composer = TagComposer(
        cluster,
        frequency,
    )

    prompt = composer.compose_pretrain_list(
        general_tags=["full body", "sitting", "solo", "watermark", "1girl"],
        copyright_tags=["vocaloid"],
        character_tags=["hatsune miku"],
        meta_tags=["watercolor (medium)", "character request", "highres"],
        rating="g",
        image_width=832,
        image_height=1152,
        temperature=1.0,
        condition_rate=0.0,
    )

    assert prompt == (
        "<|bos|>"
        "<|rating:general|><|aspect_ratio:tall|><|length:very_short|>"
        "<copyright>vocaloid</copyright>"
        "<character>hatsune miku</character>"
        "<general>1girl, solo, watercolor (medium), highres, sitting, full body</general>"
        "<|eos|>"
    )


def test_prompt_compose_sft_list():
    cluster = TagCluster.from_pretrained("data/cluster_map_1024c1.json")
    frequency = TagFrequency.from_json("data/tag_frequency.json")

    composer = TagComposer(
        cluster,
        frequency,
    )

    prompt = composer.compose_sft_list(
        general_tags=[
            "full body",
            "sitting",
            "solo",
            "watermark",
            "sample watermark",
            "too many watermarks",
            "1girl",
        ],
        copyright_tags=["vocaloid"],
        character_tags=["hatsune miku"],
        meta_tags=["watercolor (medium)", "character request", "highres"],
        rating="g",
        image_width=832,
        image_height=1152,
        temperature=1.0,
        condition_rate=0.0,
    )

    assert "watermark" not in prompt
