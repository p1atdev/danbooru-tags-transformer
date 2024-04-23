from dataclasses import dataclass

import numpy as np

from .organizer import TagOrganizer
from .rating import get_rating_tag, get_ambitious_rating_tag, SHORT_RATING_TAG
from .aspect_ratio import calculate_aspect_ratio_tag
from .length import get_length_tag
from .tags import (
    INPUT_END,
    BOS_TOKEN,
    EOS_TOKEN,
    COPYRIGHT_START,
    COPYRIGHT_END,
    CHARACTER_START,
    CHARACTER_END,
    GENERAL_START,
    GENERAL_END,
)


@dataclass
class CommonTags:
    """Common tags in the prompt."""

    length_tag: str
    rating_tag: str
    aspect_ratio_tag: str


@dataclass
class ConditionTags:
    """Condition tags in the prompt."""

    copyright_part: list[str]
    character_part: list[str]
    pre_general_part: list[str]
    post_general_part: list[str]


class TagComposer:
    organizer: TagOrganizer

    fuzzy_rating_rate: float
    drop_people_rate: float
    keep_identity_rate: float
    keep_identity_condition_rate: float
    keep_identity_token: str

    def __init__(
        self,
        organizer: TagOrganizer,
        keep_identity_token: str,
        fuzzy_rating_rate: float = 0.25,
        drop_people_rate: float = 0.1,
        keep_identity_rate: float = 0.5,
        keep_identity_condition_rate: float = 0.5,
    ):
        self.organizer = organizer
        self.keep_identity_token = keep_identity_token

        self.fuzzy_rating_rate = fuzzy_rating_rate
        self.drop_people_rate = drop_people_rate
        self.keep_identity_rate = keep_identity_rate
        self.keep_identity_condition_rate = keep_identity_condition_rate

    def recompose_tags(
        self,
        tags: list[str],
        condition_rate: float = 0.5,
    ) -> tuple[list[str], list[str]]:
        """
        Recompose tags by group and cluster.

        This keeps the identity of a character in the prompt.
        """

        result = self.organizer.organize_tags(tags)

        pre_tags = []
        post_tags = []

        pre_tags.extend(result.people_tags)
        pre_tags.extend(result.focus_tags)

        if len(result.other_tags) == 1:
            # if there is only one cluster, assign to post_tags
            post_tags.extend(result.other_tags[0])
            return pre_tags, post_tags

        for cluster_tags in result.other_tags:
            # randomly assign to pre or post
            if np.random.rand() < condition_rate:
                pre_tags.extend(cluster_tags)
            else:
                post_tags.extend(cluster_tags)

        # if post_tags or pre_tags is empty, retry
        if len(pre_tags) == 0 or len(post_tags) == 0:
            return self.recompose_tags(tags, condition_rate)

        return pre_tags, post_tags

    def get_common_tags(
        self,
        rating: SHORT_RATING_TAG,
        general: list[str],
        image_width: int,
        image_height: int,
    ):
        """Get common tags in the prompt."""

        length_tag = get_length_tag(len(general))
        rating_tag = (
            get_ambitious_rating_tag(rating)
            if np.random.rand() < self.fuzzy_rating_rate
            else get_rating_tag(rating)
        )
        aspect_ratio_tag = calculate_aspect_ratio_tag(image_width, image_height)

        return CommonTags(
            length_tag=length_tag,
            rating_tag=rating_tag,
            aspect_ratio_tag=aspect_ratio_tag,
        )

    def get_keep_identity_condition_part(
        self,
        copyright: list[str],
        character: list[str],
        general: list[str],
    ):
        # keep identity
        pre_tags, post_tags = self.recompose_tags(general)

        # shuffle pre_tags
        np.random.shuffle(pre_tags)
        # sort post_tags
        post_tags = sorted(post_tags)

        # shuffle copyright and character tags
        np.random.shuffle(copyright)
        np.random.shuffle(character)

        return ConditionTags(
            copyright_part=copyright,
            character_part=character,
            pre_general_part=pre_tags,
            post_general_part=post_tags,
        )

    def get_free_condition_part(
        self,
        copyright: list[str],
        character: list[str],
        general: list[str],
    ):
        result = self.organizer.organize_tags(general)
        other_tags = sum(result.other_tags, [])  # just flatten

        # randomly split result.other_tags
        np.random.shuffle(other_tags)
        split_index = np.random.randint(0, len(other_tags))
        pre_part = other_tags[:split_index]
        post_part = other_tags[split_index:]

        # drop people tags
        if np.random.rand() < self.drop_people_rate:
            post_part.extend(result.people_tags + result.focus_tags)
        else:
            pre_part.extend(result.people_tags + result.focus_tags)

        # shuffle pre_part
        np.random.shuffle(pre_part)
        # sort post_part
        post_part = sorted(post_part)

        # shuffle copyright and character tags
        np.random.shuffle(copyright)
        np.random.shuffle(character)

        return ConditionTags(
            copyright_part=copyright,
            character_part=character,
            pre_general_part=pre_part,
            post_general_part=post_part,
        )

    def compose_sft_prompt(
        self,
        rating: SHORT_RATING_TAG,
        copyright: list[str],
        character: list[str],
        general: list[str],
        image_width: int,
        image_height: int,
    ):
        """
        Get condition tags in the prompt.

        This function should be called when creating a dataset for SFT.
        """

        common = self.get_common_tags(
            rating=rating,
            general=general,
            image_width=image_width,
            image_height=image_height,
        )

        if np.random.rand() < self.keep_identity_condition_rate:
            condition = self.get_keep_identity_condition_part(
                copyright=copyright,
                character=character,
                general=general,
            )

            tags = [
                COPYRIGHT_START,
                ", ".join(condition.copyright_part),
                COPYRIGHT_END,
                CHARACTER_START,
                ", ".join(condition.character_part),
                CHARACTER_END,
                common.rating_tag,
                common.aspect_ratio_tag,
                common.length_tag,
                GENERAL_START,
                ", ".join(condition.pre_general_part),
                self.keep_identity_token,
                INPUT_END,
                ", ".join(condition.post_general_part),
                GENERAL_END,
            ]
        else:
            condition = self.get_free_condition_part(
                copyright=copyright,
                character=character,
                general=general,
            )

            tags = [
                COPYRIGHT_START,
                ", ".join(condition.copyright_part),
                COPYRIGHT_END,
                CHARACTER_START,
                ", ".join(condition.character_part),
                CHARACTER_END,
                common.rating_tag,
                common.aspect_ratio_tag,
                common.length_tag,
                GENERAL_START,
                ", ".join(condition.pre_general_part),
                INPUT_END,
                ", ".join(condition.post_general_part),
                GENERAL_END,
            ]

        return "".join(tags)

    def compose_prompt(
        self,
        rating: SHORT_RATING_TAG,
        copyright: list[str],
        character: list[str],
        general: list[str],
        image_width: int,
        image_height: int,
    ):
        """
        Get condition tags in the prompt.

        This function should be called when creating a dataset for pretraining.
        """

        common = self.get_common_tags(
            rating=rating,
            general=general,
            image_width=image_width,
            image_height=image_height,
        )

        # sort tags
        copyright = sorted(copyright)
        character = sorted(character)
        general = sorted(general)

        tags = [
            COPYRIGHT_START,
            ", ".join(copyright),
            COPYRIGHT_END,
            CHARACTER_START,
            ", ".join(character),
            CHARACTER_END,
            common.rating_tag,
            common.aspect_ratio_tag,
            common.length_tag,
            GENERAL_START,
            ", ".join(general),
            GENERAL_END,
        ]

        return "".join(tags)
