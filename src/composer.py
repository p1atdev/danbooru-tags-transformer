import math

from enum import Enum
from typing import Tuple
from dataclasses import dataclass
import json
import random

import numpy as np

from .rating import get_rating_tag, get_ambitious_rating_tag, SHORT_RATING_TAG
from .aspect_ratio import calculate_aspect_ratio_tag, is_extreme_aspect_ratio
from .length import get_length_tag, is_too_long_tags
from .cluster import TagCluster
from .formatter import (
    format_completion,
    format_sft_with_initial_condition,
    format_sft_with_use_condition,
    format_ndart_with_simple_conversion,
)


def random_choose(tags: list[str], rate: float) -> tuple[list[str], list[str]]:
    """
    Tiven a list of tags, randomly choose tags.
    """
    chosen = []
    remains = []
    for tag in tags:
        if random.random() < rate:
            chosen.append(tag)
        else:
            remains.append(tag)
    return chosen, remains


class PredefinedTagType(Enum):
    # 含まれていたらそのデータ全体を使わない
    BAN = "ban"
    # 削除
    REMOVE = "remove"
    # 先頭に挿入
    INSERT_START = "insert_start"
    # 保持
    KEEP = "keep"


class MatchingType(Enum):
    INCLUDE = "include"  # 部分一致
    FULL = "full"  # 完全一致


class PredefinedTags:
    """
    事前に指定したタグの管理をおこなうクラス
    """

    # タグのリスト
    tags: list[str]
    tag_type: PredefinedTagType

    def __init__(
        self, tags: list[str], tag_type: PredefinedTagType, matching_type: MatchingType
    ):
        self.tags = tags
        self.tag_type = tag_type
        self.matching_type = matching_type

    # テキストファイルから読み込む
    @classmethod
    def from_txt_file(
        cls, path: str, tag_type: PredefinedTagType, matching_type: MatchingType
    ) -> "PredefinedTags":
        with open(path, "r") as f:
            tags = f.readlines()
            tags = [tag.strip() for tag in tags if tag.strip()]
        return cls(tags, tag_type, matching_type)

    @classmethod
    def artistic_error(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/artistic_error.txt",
            PredefinedTagType.REMOVE,
            MatchingType.INCLUDE,
        )

    @classmethod
    def background(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/background.txt",
            PredefinedTagType.KEEP,
            MatchingType.FULL,
        )

    @classmethod
    def ban_meta(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/ban_meta.txt",
            PredefinedTagType.BAN,
            MatchingType.INCLUDE,
        )

    @classmethod
    def color_theme(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/color_theme.txt",
            PredefinedTagType.KEEP,
            MatchingType.FULL,
        )

    @classmethod
    def displeasing_meta(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/displeasing_meta.txt",
            PredefinedTagType.REMOVE,
            MatchingType.INCLUDE,
        )

    @classmethod
    def focus(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/focus.txt",
            PredefinedTagType.KEEP,
            MatchingType.FULL,
        )

    @classmethod
    def people(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/people.txt",
            PredefinedTagType.KEEP,
            MatchingType.FULL,
        )

    @classmethod
    def usable_meta(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/usable_meta.txt",
            PredefinedTagType.INSERT_START,
            MatchingType.FULL,
        )

    @classmethod
    def medium(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/medium.txt",
            PredefinedTagType.INSERT_START,
            MatchingType.FULL,
        )

    @classmethod
    def watermark(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/watermark.txt",
            PredefinedTagType.REMOVE,
            MatchingType.FULL,
        )

    @classmethod
    def condition_only(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/condition_only.txt",
            PredefinedTagType.KEEP,
            MatchingType.FULL,
        )

    @classmethod
    def text(cls) -> "PredefinedTags":
        return cls.from_txt_file(
            "tags/text.txt",
            PredefinedTagType.KEEP,
            MatchingType.FULL,
        )


# 出現頻度を計算するクラス
class TagFrequency:
    # タグ名と出現回数の辞書
    tag_to_frequency: dict[str, int]

    def __init__(self, tag_to_frequency: dict[str, int]):
        self.tag_to_frequency = tag_to_frequency

    # テキストファイルから読み込む
    @classmethod
    def from_json(cls, path: str):
        with open(path, "r") as f:
            tag_to_frequency = json.load(f)
        return cls(tag_to_frequency)


# 平均取ってからsoftmaxを計算する
def avg_softmax(logits: np.ndarray, temperature: float = 1.0):
    logits = logits / np.sum(logits)
    logits = logits / temperature
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    result = exps / np.sum(exps)
    return result


# タグの順序を整理するクラス
class TagSelector:
    # タグ名と最適な絶対位置インデックスの辞書
    tag_to_position: dict[str, int]

    # タグ名とクラスターidとクラスター内の位置
    tag_to_cluster_position: dict[str, Tuple[int, int]]

    # 手動で設定するタグのリスト
    high_priority_groups: list[PredefinedTags]

    # クラスターidとタグのリスト
    cluster_list: dict[int, list[str]]

    # クラスターidと登場回数順位
    cluster_ranking: dict[int, int]

    # 登場回数情報
    frequency: TagFrequency

    def __init__(
        self,
        cluster: TagCluster,
        frequency: TagFrequency,
        high_priority_groups: list[PredefinedTags],
    ):
        self.cluster_list = {}
        # クラスターごとにタグを分類
        for tag, cluster_id in cluster.cluster_map.items():
            if cluster_id not in self.cluster_list:
                self.cluster_list[cluster_id] = []
            if tag in frequency.tag_to_frequency:
                self.cluster_list[cluster_id].append(tag)

        self.high_priority_groups = high_priority_groups
        self.frequency = frequency

        self.calculate_rankings()

    def cluster_id_to_tags(self, cluster_id: int) -> list[str]:
        return self.cluster_list[cluster_id]

    def calculate_rankings(self):
        # cluster id to tag list
        cluster_max_freq = {}

        # クラスターごとの平均出現頻度を計算
        for cluster_id, tags in self.cluster_list.items():
            if len(tags) == 0:
                cluster_max_freq[cluster_id] = -1
                continue

            cluster_max_freq[cluster_id] = sum(
                [self.frequency.tag_to_frequency[tag] for tag in tags]
            ) / len(tags)
        # 出現頻度が高い順にソート
        cluster_max_freq = dict(sorted(cluster_max_freq.items(), key=lambda x: -x[1]))

        # クラスターidと平均出現頻度の順位を計算
        self.cluster_ranking = {
            cluster_id: i for i, (cluster_id, _) in enumerate(cluster_max_freq.items())
        }

        # クラスター内でタグの順位を計算
        in_cluster_ranking = {}
        self.tag_to_cluster_position = {}
        for cluster_id, tags in self.cluster_list.items():
            tags = sorted(
                tags, key=lambda x: -self.frequency.tag_to_frequency[x]
            )  # 出現頻度が高い順にソート
            for i, tag in enumerate(tags):
                # クラスター内のタグの順位を計算
                in_cluster_ranking[tag] = i

                # クラスターidとクラスター内の位置を記録
                self.tag_to_cluster_position[tag] = (cluster_id, i)

        # タグの絶対位置を計算
        self.tag_to_position = {}
        sorted_cluster_id_to_position = sorted(
            self.cluster_ranking.items(), key=lambda x: x[1]
        )
        for [cluster_id, cluster_position] in sorted_cluster_id_to_position:
            cluster_tags = self.cluster_list[cluster_id]
            sorted_cluster_tags = sorted(
                cluster_tags, key=lambda x: in_cluster_ranking[x]
            )
            for i, tag in enumerate(sorted_cluster_tags):
                self.tag_to_position[tag] = i + cluster_position

    # タグのリストを受け取って、クラスターidとタグのリストの辞書を返す
    def clustering_tags(self, tags: list[str]) -> dict[int, list[str]]:
        result: dict[int, list[str]] = {}
        for tag in tags:
            cluster_id, _ = self.tag_to_cluster_position.get(tag, (-1, -1))
            if cluster_id == -1:
                print(f"Warning: Tag {tag} is not found in the cluster")
                continue
            if cluster_id not in result:
                result[cluster_id] = []
            result[cluster_id].append(tag)

        return result

    # タグのリストを受け取って、そのタグの登場回数を返す
    def get_frequencies(self, tags: list[str]) -> list[int]:
        freqs = [self.frequency.tag_to_frequency[tag] for tag in tags]
        return freqs

    # 指定割合で条件に入れるタグを分類する
    def random_conditioning(
        self,
        tags: list[str],
        condition_rate: float = 0.8,
        # temperature: float = 1.0,
    ) -> Tuple[list[str], list[str]]:
        # all_freqs = self.get_frequencies(tags)

        if condition_rate == 0:
            # 条件なしならそのまま返す
            return [], tags

        if len(tags) == 0:
            raise ValueError("tags must not be empty")

        max_condition_tags = math.ceil(len(tags) * condition_rate)
        condition_tags_count = min(random.randint(0, max_condition_tags), len(tags) - 1)

        # randomly take condition tags
        if condition_tags_count == 0:
            return [], tags  # no condition tags

        conditions = random.sample(tags, condition_tags_count)
        others = [tag for tag in tags if tag not in conditions]

        if len(conditions) == 0 and len(others) == 0:  # 一度も条件が発生しなかった場合
            others = tags

        # 抜けがないかチェック
        assert len(conditions) + len(others) == len(
            tags
        ), f"{len(conditions)} + {len(others)} != {len(tags)}"

        return conditions, others

    # 出現頻度が低いタグを取り除く
    def remove_low_frequency_tags(
        self,
        tags: list[str],
        threshold: int = 100,
    ) -> list[str]:
        return [tag for tag in tags if self.frequency.tag_to_frequency[tag] > threshold]

    # 先約のみを分離
    def separate_high_priority_tags(
        self,
        tags: list[str],
    ) -> tuple[list[list[str]], list[str]]:
        high_priority_tags: list[list[str]] = []
        low_priority_tags: list[str] = self.remove_low_frequency_tags(tags)

        for i, group in enumerate(self.high_priority_groups):
            high_priority_tags.append([])

            for tag in low_priority_tags.copy():
                if group.matching_type == MatchingType.INCLUDE:
                    for group_tag in group.tags:
                        if group_tag in tag:
                            high_priority_tags[i].append(tag)
                            low_priority_tags.remove(tag)
                            break  # groupのチェックを終わる
                elif group.matching_type == MatchingType.FULL:
                    if tag in group.tags:
                        high_priority_tags[i].append(tag)
                        low_priority_tags.remove(tag)
                else:
                    raise ValueError("Invalid matching type")

        return high_priority_tags, low_priority_tags

    # 単純に出現頻度順にソートする
    def sort_tags_by_frequency(
        self,
        tags: list[str],
    ) -> list[str]:
        return sorted(tags, key=lambda x: self.frequency.tag_to_frequency[x])

    # 事前計算した絶対位置順にソートする
    def sort_tags_by_position(
        self,
        tags: list[str],
    ) -> list[str]:
        return sorted(tags, key=lambda x: self.tag_to_position[x])


# タグのプロンプトを生成するクラス
class TagComposer:
    cluster: TagCluster
    frequency: TagFrequency
    selector: TagSelector

    predefined_meta_tags: list[PredefinedTags] = [
        PredefinedTags.ban_meta(),
        PredefinedTags.displeasing_meta(),
        PredefinedTags.usable_meta(),
        PredefinedTags.medium(),
    ]
    predefined_general_tags: list[PredefinedTags] = [
        PredefinedTags.artistic_error(),
        PredefinedTags.watermark(),
        PredefinedTags.people(),
        PredefinedTags.focus(),
        PredefinedTags.color_theme(),
        PredefinedTags.background(),
        # ↓ 除外しないが、生成部分には入れないタグ
        PredefinedTags.condition_only(),  # comic など、条件部分に必ず入るタグ
        PredefinedTags.text(),  # english text など
    ]

    def __init__(self, cluster: TagCluster, frequency: TagFrequency):
        self.cluster = cluster
        self.frequency = frequency

        self.general_selector = self.get_selector(self.predefined_general_tags)
        self.meta_selector = self.get_selector(self.predefined_meta_tags)

    def get_selector(self, predefined: list[PredefinedTags]):
        return TagSelector(self.cluster, self.frequency, predefined)

    def compose_pretrain_list(
        self,
        general_tags: list[str],
        copyright_tags: list[str],
        character_tags: list[str],
        meta_tags: list[str],
        rating: SHORT_RATING_TAG,
        image_width: int,
        image_height: int,
    ) -> str | None:  # returns None if the prompt should be skipped
        # タグを取得
        if is_extreme_aspect_ratio(image_width, image_height):
            return None
        aspect_ratio_tag = calculate_aspect_ratio_tag(image_width, image_height)

        # ほかのタグ
        rating_tag = get_rating_tag(rating)
        length_tag = get_length_tag(len(general_tags))

        assert isinstance(general_tags, list)
        if len(general_tags) == 0:
            return None

        # タグをソート
        high_priortiy_general, low_priority_general = (
            self.general_selector.separate_high_priority_tags(
                general_tags,
            )
        )
        high_priority_meta, low_priority_meta = (
            self.meta_selector.separate_high_priority_tags(meta_tags)
        )

        # 条件パート | 生成パート
        keep_meta_part = []  # 絶対に条件になる meta
        keep_general_part = []  # 絶対に条件になる general
        insert_meta_part = []  # 先約あり meta、シャッフルできない、ソートして配置
        insert_general_part = []  # 先約あり general、シャッフルできない、ソートして配置
        meta_part = low_priority_meta  # 生成部分、シャッフルしない、ソートして配置
        general_part = (
            low_priority_general  # 生成部分, シャッフルしない、ソートして配置
        )

        ## 1. 事前定義したタグかどうか

        for tags, predefined in zip(
            high_priortiy_general,
            self.general_selector.high_priority_groups,
            strict=True,
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                insert_general_part.extend(tags)  # 優先枠
            elif predefined.tag_type == PredefinedTagType.KEEP:
                keep_general_part.extend(tags)  # 確定枠

        for tags, predefined in zip(
            high_priority_meta,
            self.meta_selector.high_priority_groups,
            strict=True,
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                insert_meta_part.extend(tags)  # 優先枠
            elif predefined.tag_type == PredefinedTagType.KEEP:
                keep_meta_part.extend(tags)  # 確定枠

        ## 2. ソート

        generation_part = (
            self.general_selector.sort_tags_by_position(keep_general_part)  # 確定枠先に
            # generalタグは事前計算した絶対位置順にソート
            + self.general_selector.sort_tags_by_position(insert_general_part)
            + self.meta_selector.sort_tags_by_frequency(
                keep_meta_part + insert_meta_part
            )
            # + self.meta_selector.sort_tags_by_frequency(meta_part)
            + self.general_selector.sort_tags_by_position(general_part)
        )

        # 出現頻度順にソート
        character_tags = self.general_selector.sort_tags_by_frequency(character_tags)
        copyright_tags = self.general_selector.sort_tags_by_frequency(copyright_tags)

        # テンプレートに適用
        prompt = format_completion(
            generation=generation_part,
            character=character_tags,
            copyright=copyright_tags,
            rating=rating_tag,
            aspect_ratio=aspect_ratio_tag,
            length=length_tag,
        )

        return prompt

    def compose_sft_list(
        self,
        general_tags: list[str],
        copyright_tags: list[str],
        character_tags: list[str],
        meta_tags: list[str],
        rating: SHORT_RATING_TAG,
        image_width: int,
        image_height: int,
        condition_rate: float = 0.0,
        full_dropout_rate: float = 0.05,  # 5%の確率で全てのgeneralタグをドロップして条件に含めない
        original_dropout_rate: float = 0.5,  # originalタグをドロップする確率
    ) -> str | None:  # returns None if the prompt should be skipped
        # タグを取得
        if is_extreme_aspect_ratio(image_width, image_height):
            return None
        aspect_ratio_tag = calculate_aspect_ratio_tag(image_width, image_height)

        # ほかのタグ
        rating_tag = get_rating_tag(rating)
        length_tag = get_length_tag(len(general_tags))

        assert isinstance(general_tags, list)
        if len(general_tags) == 0:
            return None

        is_full_dropout = full_dropout_rate > 0 and random.random() < full_dropout_rate

        # 先約済み、残り
        high_priortiy_general, low_priority_general = (
            self.general_selector.separate_high_priority_tags(
                general_tags,
            )
        )
        if len(low_priority_general) == 0:
            # 生成部分がないなら削除
            return None
        high_priority_meta, low_priority_meta = (
            self.meta_selector.separate_high_priority_tags(meta_tags)
        )

        # 条件パート | 生成パート
        keep_meta_part = []  # 絶対に条件になる meta
        keep_general_part = []  # 絶対に条件になる general
        insert_meta_part = []  # 先約あり meta、シャッフルできない、ソートして配置
        insert_general_part = []  # 先約あり general、シャッフルできない、ソートして配置
        meta_part = low_priority_meta  # 生成部分、シャッフルしない、ソートして配置
        general_part = (
            low_priority_general  # 生成部分, シャッフルしない、ソートして配置
        )

        ## 1. 事前定義したタグかどうか

        for tags, predefined in zip(
            high_priortiy_general,
            self.general_selector.high_priority_groups,
            strict=True,
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                insert_general_part.extend(tags)  # 優先枠
            elif predefined.tag_type == PredefinedTagType.KEEP:
                keep_general_part.extend(tags)  # 確定枠

        for tags, predefined in zip(
            high_priority_meta,
            self.meta_selector.high_priority_groups,
            strict=True,
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                insert_meta_part.extend(tags)  # 優先枠
            elif predefined.tag_type == PredefinedTagType.KEEP:
                keep_meta_part.extend(tags)  # 確定枠

        ## 2. 条件部分の作成
        condition_part = []
        generation_part = []
        if is_full_dropout:
            # 条件部分を全部補完側に
            condition_part = []  # 条件なし
            # 個別にソート
            generation_part = (
                self.general_selector.sort_tags_by_position(keep_general_part)
                + self.general_selector.sort_tags_by_position(insert_general_part)
                + self.meta_selector.sort_tags_by_frequency(
                    keep_meta_part + insert_meta_part  # + meta_part
                )
                + self.general_selector.sort_tags_by_position(general_part)
            )
        else:
            condition_general, generation_general = random_choose(
                general_part, condition_rate
            )
            # condition_meta, generation_meta = random_choose(meta_part, condition_rate)
            insert_condition_general, insert_generation_general = random_choose(
                insert_general_part, condition_rate
            )
            insert_condition_meta, insert_generation_meta = random_choose(
                insert_meta_part, condition_rate
            )
            condition_part = (
                keep_general_part
                + keep_meta_part
                + condition_general
                # + condition_meta
                + insert_condition_meta
                + insert_condition_general
            )
            generation_part = (
                self.meta_selector.sort_tags_by_frequency(insert_generation_meta)
                + self.general_selector.sort_tags_by_position(insert_generation_general)
                # + self.meta_selector.sort_tags_by_frequency(generation_meta)
                + self.general_selector.sort_tags_by_position(generation_general)
            )

        # オリジナルなら original タグを確率でドロップ
        if copyright_tags == ["original"] and character_tags == []:
            if random.random() < original_dropout_rate:  # 50%の確率でドロップ
                copyright_tags = []

        # シャッフル
        random.shuffle(condition_part)
        random.shuffle(character_tags)
        random.shuffle(copyright_tags)

        # まとめてシャッフル
        rating_aspect_ratio_length = [rating_tag, aspect_ratio_tag, length_tag]
        random.shuffle(rating_aspect_ratio_length)

        # テンプレートに適用
        prompt = format_sft_with_initial_condition(
            rating_aspect_ratio_length=rating_aspect_ratio_length,  # shuffled
            condition=condition_part,  # shuffled
            copyright=copyright_tags,  # shuffled
            character=character_tags,  # shuffled
            generation=generation_part,
        )

        return prompt

    def compose_sft_use_list(
        self,
        general_tags: list[str],
        copyright_tags: list[str],
        character_tags: list[str],
        meta_tags: list[str],
        rating: SHORT_RATING_TAG,
        image_width: int,
        image_height: int,
        condition_rate: float = 0.0,
        full_dropout_rate: float = 0.05,  # 5%の確率で全てのgeneralタグをドロップして条件に含めない
        original_dropout_rate: float = 0.5,  # originalタグをドロップする確率
    ) -> str | None:  # returns None if the prompt should be skipped
        # タグを取得
        if is_extreme_aspect_ratio(image_width, image_height):
            return None
        aspect_ratio_tag = calculate_aspect_ratio_tag(image_width, image_height)

        # ほかのタグ
        rating_tag = get_rating_tag(rating)
        length_tag = get_length_tag(len(general_tags))

        assert isinstance(general_tags, list)
        if len(general_tags) == 0:
            return None

        is_full_dropout = full_dropout_rate > 0 and random.random() < full_dropout_rate

        # 先約済み、残り
        high_priortiy_general, low_priority_general = (
            self.general_selector.separate_high_priority_tags(
                general_tags,
            )
        )
        if len(low_priority_general) == 0:
            # 生成部分がないなら削除
            return None
        high_priority_meta, low_priority_meta = (
            self.meta_selector.separate_high_priority_tags(meta_tags)
        )

        # 条件パート | 生成パート
        keep_meta_part = []  # 絶対に条件になる meta
        keep_general_part = []  # 絶対に条件になる general
        insert_meta_part = []  # 先約あり meta、シャッフルできない、ソートして配置
        insert_general_part = []  # 先約あり general、シャッフルできない、ソートして配置
        meta_part = low_priority_meta  # 生成部分、シャッフルしない、ソートして配置
        general_part = (
            low_priority_general  # 生成部分, シャッフルしない、ソートして配置
        )

        ## 1. 事前定義したタグかどうか

        for tags, predefined in zip(
            high_priortiy_general,
            self.general_selector.high_priority_groups,
            strict=True,
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                insert_general_part.extend(tags)  # 優先枠
            elif predefined.tag_type == PredefinedTagType.KEEP:
                keep_general_part.extend(tags)  # 確定枠

        for tags, predefined in zip(
            high_priority_meta,
            self.meta_selector.high_priority_groups,
            strict=True,
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                insert_meta_part.extend(tags)  # 優先枠
            elif predefined.tag_type == PredefinedTagType.KEEP:
                keep_meta_part.extend(tags)  # 確定枠

        ## 2. 条件部分の作成
        condition_part = []
        generation_part = []
        if is_full_dropout:
            # 条件部分を全部補完側に
            condition_part = []  # 条件なし
            # 個別にソート
            generation_part = (
                self.general_selector.sort_tags_by_position(keep_general_part)
                + self.general_selector.sort_tags_by_position(insert_general_part)
                + self.meta_selector.sort_tags_by_frequency(
                    keep_meta_part + insert_meta_part  # +meta_part
                )
                + self.general_selector.sort_tags_by_position(general_part)
            )
        else:
            condition_general, generation_general = random_choose(
                general_part, condition_rate
            )
            # condition_meta, generation_meta = random_choose(meta_part, condition_rate)
            insert_condition_general, insert_generation_general = random_choose(
                insert_general_part, condition_rate
            )
            insert_condition_meta, insert_generation_meta = random_choose(
                insert_meta_part, condition_rate
            )
            condition_part = (
                keep_general_part
                + keep_meta_part
                + condition_general
                # + condition_meta
                + insert_condition_meta
                + insert_condition_general
            )
            generation_part = (
                self.meta_selector.sort_tags_by_frequency(insert_generation_meta)
                + self.general_selector.sort_tags_by_position(insert_generation_general)
                # + self.meta_selector.sort_tags_by_frequency(generation_meta)
                + self.general_selector.sort_tags_by_position(generation_general)
            )

        # オリジナルなら original タグを確率でドロップ
        if copyright_tags == ["original"] and character_tags == []:
            if random.random() < original_dropout_rate:  # 50%の確率でドロップ
                copyright_tags = []

        # 生成部分
        # condition_tags をシャッフルする前に取得
        generation_part = ["<group>"] + condition_part + ["</group>"] + general_part

        # # シャッフル
        random.shuffle(condition_part)
        random.shuffle(character_tags)
        random.shuffle(copyright_tags)

        # まとめてシャッフル
        rating_aspect_ratio_length = [rating_tag, aspect_ratio_tag, length_tag]
        random.shuffle(rating_aspect_ratio_length)

        assert len(condition_part) != len(generation_part)

        # テンプレートに適用
        prompt = format_sft_with_use_condition(
            rating_aspect_ratio_length=rating_aspect_ratio_length,  # shuffled
            condition=condition_part,  # shuffled
            copyright=copyright_tags,  # shuffled
            character=character_tags,  # shuffled
            meta_general=generation_part,
        )

        return prompt

    def compose_ndart_list(
        self,
        general_tags: list[str],
        copyright_tags: list[str],
        character_tags: list[str],
        meta_tags: list[str],
        rating: SHORT_RATING_TAG,
        image_width: int,
        image_height: int,
        condition_rate: float = 0.0,
    ) -> str | None:  # returns None if the prompt should be skipped
        # タグを取得
        if is_extreme_aspect_ratio(image_width, image_height):
            return None
        aspect_ratio_tag = calculate_aspect_ratio_tag(image_width, image_height)

        # ほかのタグ
        rating_tag = get_rating_tag(rating)
        length_tag = get_length_tag(len(general_tags))

        assert isinstance(general_tags, list)
        if len(general_tags) == 0:
            return None

        # 先約済み、残り
        high_priortiy_general, low_priority_general = (
            self.general_selector.separate_high_priority_tags(
                general_tags,
            )
        )
        if len(low_priority_general) == 0:
            # 生成部分がないなら削除
            return None
        high_priority_meta, low_priority_meta = (
            self.meta_selector.separate_high_priority_tags(meta_tags)
        )

        # 条件パート | 生成パート
        keep_meta_part = []  # 絶対に条件になる meta
        keep_general_part = []  # 絶対に条件になる general
        insert_meta_part = []  # 先約あり meta、シャッフルできない、ソートして配置
        insert_general_part = []  # 先約あり general、シャッフルできない、ソートして配置
        meta_part = low_priority_meta  # 生成部分、シャッフルしない、ソートして配置
        general_part = (
            low_priority_general  # 生成部分, シャッフルしない、ソートして配置
        )

        ## 1. 事前定義したタグかどうか

        for tags, predefined in zip(
            high_priortiy_general,
            self.general_selector.high_priority_groups,
            strict=True,
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                insert_general_part.extend(tags)  # 優先枠
            elif predefined.tag_type == PredefinedTagType.KEEP:
                keep_general_part.extend(tags)  # 確定枠

        for tags, predefined in zip(
            high_priority_meta,
            self.meta_selector.high_priority_groups,
            strict=True,
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                insert_meta_part.extend(tags)  # 優先枠
            elif predefined.tag_type == PredefinedTagType.KEEP:
                keep_meta_part.extend(tags)  # 確定枠

        ## 2. ソート

        generation_part = (
            self.general_selector.sort_tags_by_position(keep_general_part)  # 確定枠先に
            # generalタグは事前計算した絶対位置順にソート
            + self.general_selector.sort_tags_by_position(insert_general_part)
            + self.meta_selector.sort_tags_by_frequency(
                keep_meta_part + insert_meta_part
            )
            # + self.meta_selector.sort_tags_by_frequency(meta_part)
            + self.general_selector.sort_tags_by_position(general_part)
        )

        rating_aspect_ratio_length = [rating_tag, aspect_ratio_tag, length_tag]

        ## 3. シャッフル
        random.shuffle(rating_aspect_ratio_length)
        random.shuffle(character_tags)
        random.shuffle(copyright_tags)

        # テンプレートに適用
        prompt = format_ndart_with_simple_conversion(
            rating_aspect_ratio_length=rating_aspect_ratio_length,
            copyright=copyright_tags,
            character=character_tags,
            generation=generation_part,
        )

        return prompt

    # 自然言語の入力のフォーマット
    def compose_natural_list(
        self,
        general_tags: list[str],
        copyright_tags: list[str],
        character_tags: list[str],
    ):
        tags = general_tags + character_tags + copyright_tags
        random.shuffle(tags)

        return ", ".join(tags)
