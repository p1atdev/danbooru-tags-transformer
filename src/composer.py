import math

from dataclasses import dataclass

import numpy as np

from .organizer import TagOrganizerResult
from .rating import get_rating_tag, get_ambitious_rating_tag, SHORT_RATING_TAG
from .aspect_ratio import calculate_aspect_ratio_tag
from .length import get_length_tag


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


@dataclass
class PrompotComponents:
    """Prompt components."""

    copyright: str
    character: str
    general_condition: str
    general_completion: str
    rating: str
    aspect_ratio: str
    length: str


class TagComposer:
    # rating タグをあいまいにする確率
    fuzzy_rating_tag_rate: float
    # 人物タグを条件部分に入れ**ない**確率
    drop_people_rate: float

    # それぞれのクラスターかタグが条件グループに入る確率
    condition_rate: float
    # 版権タグ、キャラクタータグがついているときに、条件とするタグの数を減らす確率
    copyright_character_augmentation_rate: float

    def __init__(
        self,
        fuzzy_rating_tag_rate: float = 0.25,
        drop_people_rate: float = 0.1,
        condition_rate: float = 0.5,
        copyright_character_augmentation_rate: float = 1.25,
    ):

        self.fuzzy_rating_tag_rate = fuzzy_rating_tag_rate
        self.drop_people_rate = drop_people_rate

        self.condition_rate = condition_rate
        self.copyright_character_augmentation_rate = (
            copyright_character_augmentation_rate
        )

    def recompose_tags(
        self,
        organizer_result: TagOrganizerResult,
        condition_augmentation_rate: float = 1.0,  # 条件グループへの入りやすさ
    ) -> tuple[list[str], list[str]]:
        """
        Recompose tags by group and cluster.

        This keeps the identity of a character in the prompt.
        """

        # result = self.organizer.organize_tags(tags)

        pre_tags = []
        post_tags = []

        pre_tags.extend(organizer_result.people_tags)
        pre_tags.extend(organizer_result.focus_tags)

        if len(organizer_result.other_tags) == 1:
            # if there is only one cluster, randomly assign to post_tags
            for tag in organizer_result.other_tags[0]:
                if np.random.rand() < self.condition_rate:
                    pre_tags.append(tag)
                else:
                    post_tags.append(tag)
            return pre_tags, post_tags

        for cluster_tags in organizer_result.other_tags:
            # randomly assign to pre or post
            if (
                np.random.rand()
                < self.condition_rate
                / condition_augmentation_rate  # 条件グループに含める確率を上げる
            ):
                pre_tags.extend(cluster_tags)
            else:
                post_tags.extend(cluster_tags)

        # if post_tags is empty, set pre_tags to post_tags
        if len(post_tags) == 0:
            post_tags = pre_tags
            pre_tags = []

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
            if np.random.rand() < self.fuzzy_rating_tag_rate
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
        organizer_result: TagOrganizerResult,
    ):
        # なにかしらの版権テーマ
        is_copyright_character = len(
            [tag for tag in copyright if tag != "original"]
        ) > 0 and len(character)

        # keep identity
        pre_tags, post_tags = self.recompose_tags(
            organizer_result,
            (
                # 版権かキャラクタータグがあれば条件部分のタグ数を減らす
                self.copyright_character_augmentation_rate
                if is_copyright_character
                else 1.0
            ),
        )

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
        organizer_result: TagOrganizerResult,
    ):
        other_tags = sum(organizer_result.other_tags, [])  # just flatten

        # なにかしらの版権テーマ
        is_copyright_character = len(
            [tag for tag in copyright if tag != "original"]
        ) > 0 and len(character)

        # randomly split result.other_tags
        np.random.shuffle(other_tags)
        split_index = np.random.randint(
            0,
            math.ceil(
                len(other_tags)
                # 条件部分に入る確率
                * self.condition_rate
                # 版権的なものであれば条件部分のタグ数を減らす
                / (
                    self.copyright_character_augmentation_rate
                    if is_copyright_character
                    else 1.0
                )
            ),
        )
        pre_part = other_tags[:split_index]
        post_part = other_tags[split_index:]

        # drop people tags
        if np.random.rand() < self.drop_people_rate:
            post_part.extend(organizer_result.people_tags + organizer_result.focus_tags)
        else:
            pre_part.extend(organizer_result.people_tags + organizer_result.focus_tags)

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

    def get_components_identity_keep(
        self,
        rating: SHORT_RATING_TAG,
        copyright: list[str],
        character: list[str],
        organizer_result: TagOrganizerResult,
        image_width: int,
        image_height: int,
    ):
        """Get prompt components with keeping identity."""

        common = self.get_common_tags(
            rating=rating,
            general=organizer_result.people_tags
            + organizer_result.focus_tags
            + sum(organizer_result.other_tags, []),  # just flatten,
            image_width=image_width,
            image_height=image_height,
        )

        condition = self.get_keep_identity_condition_part(
            copyright=copyright,
            character=character,
            organizer_result=organizer_result,
        )

        return PrompotComponents(
            copyright=", ".join(condition.copyright_part),
            character=", ".join(condition.character_part),
            general_condition=", ".join(condition.pre_general_part),
            general_completion=", ".join(condition.post_general_part),
            rating=common.rating_tag,
            aspect_ratio=common.aspect_ratio_tag,
            length=common.length_tag,
        )

    def get_components_identity_free(
        self,
        rating: SHORT_RATING_TAG,
        copyright: list[str],
        character: list[str],
        organizer_result: TagOrganizerResult,
        image_width: int,
        image_height: int,
    ):
        """Get prompt components without keeping identity."""

        common = self.get_common_tags(
            rating=rating,
            general=organizer_result.people_tags
            + organizer_result.focus_tags
            + sum(organizer_result.other_tags, []),  # just flatten
            image_width=image_width,
            image_height=image_height,
        )

        condition = self.get_free_condition_part(
            copyright=copyright,
            character=character,
            organizer_result=organizer_result,
        )

        return PrompotComponents(
            copyright=", ".join(condition.copyright_part),
            character=", ".join(condition.character_part),
            general_condition=", ".join(condition.pre_general_part),
            general_completion=", ".join(condition.post_general_part),
            rating=common.rating_tag,
            aspect_ratio=common.aspect_ratio_tag,
            length=common.length_tag,
        )

    def get_components(
        self,
        rating: SHORT_RATING_TAG,
        copyright: list[str],
        character: list[str],
        organizer_result: TagOrganizerResult,
        image_width: int,
        image_height: int,
    ):
        """Get prompt components."""

        general = (
            organizer_result.people_tags
            + organizer_result.focus_tags
            + sum(organizer_result.other_tags, [])
        )

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

        return PrompotComponents(
            copyright=", ".join(copyright),
            character=", ".join(character),
            general_condition="",
            general_completion=", ".join(general),
            rating=common.rating_tag,
            aspect_ratio=common.aspect_ratio_tag,
            length=common.length_tag,
        )
