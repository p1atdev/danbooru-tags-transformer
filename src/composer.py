import math

from enum import Enum
from typing import Tuple
from dataclasses import dataclass
import json

import numpy as np

from .organizer import TagOrganizerResult
from .rating import get_rating_tag, get_ambitious_rating_tag, SHORT_RATING_TAG
from .aspect_ratio import calculate_aspect_ratio_tag
from .length import get_length_tag
from .cluster import TagCluster


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


class PredefinedTagType(Enum):
    # 含まれていたらそのデータ全体を使わない
    BAN = "ban"
    # 削除
    REMOVE = "remove"
    # 先頭に挿入
    INSERT_START = "insert_start"


class PredefinedTags:
    """
    事前に指定したタグの管理をおこなうクラス
    """

    # タグのリスト
    tags: list[str]

    def __init__(self, tags: list[str]):
        self.tags = tags

    # テキストファイルから読み込む
    @classmethod
    def from_txt_file(cls, path: str):
        with open(path, "r") as f:
            tags = f.readlines()
            tags = [tag.strip() for tag in tags if tag.strip()]
        return cls(tags)


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


# タグの順序を整理するクラス
class TagSorter:
    # タグ名と最適な絶対位置インデックスの辞書
    tag_to_position: dict[str, int]

    # 手動で設定するタグのリスト
    high_priority_groups: list[PredefinedTags]

    def __init__(
        self,
        cluster: TagCluster,
        frequency: TagFrequency,
        high_priority_groups: list[PredefinedTags],
    ):
        self.high_priority_groups = high_priority_groups
        self.tag_to_position = self.calculate_rankings(cluster, frequency)

    def calculate_rankings(self, cluster: TagCluster, frequency: TagFrequency):
        cluster_list: dict[int, list[str]] = {}
        cluster_avg_freq: dict[int, float] = {}

        # クラスターごとにタグを分類
        for tag, cluster_id in cluster.cluster_map.items():
            if cluster_id not in cluster_list:
                cluster_list[cluster_id] = []
            if tag in frequency.tag_to_frequency:
                cluster_list[cluster_id].append(tag)

        # クラスターごとの平均出現頻度を計算
        for cluster_id, tags in cluster_list.items():
            if len(tags) == 0:
                cluster_avg_freq[cluster_id] = -1
                continue

            cluster_avg_freq[cluster_id] = sum(
                [frequency.tag_to_frequency[tag] for tag in tags]
            ) / len(tags)

        # クラスターidと平均出現頻度の順位を計算
        cluster_ranking: dict[int, int] = {}
        sorted_cluster_avg_freq = sorted(cluster_avg_freq.values(), key=lambda x: -x)
        for cluster_id in cluster_avg_freq.keys():
            cluster_ranking[cluster_id] = sorted_cluster_avg_freq.index(
                cluster_avg_freq[cluster_id]
            )

        # クラスター内でタグの順位を計算
        in_cluster_ranking: dict[str, int] = {}
        for cluster_id, tags in cluster_list.items():
            tags = sorted(
                tags, key=lambda x: -frequency.tag_to_frequency[x]
            )  # 出現頻度が高い順にソート
            for i, tag in enumerate(tags):
                # クラスター内のタグの順位を計算
                in_cluster_ranking[tag] = i

        # タグの順位を計算
        tag_to_position = {}
        sorted_cluster_id_to_position = sorted(
            cluster_ranking.items(), key=lambda x: x[1]
        )
        for [cluster_id, position] in sorted_cluster_id_to_position:
            cluster_tags = cluster_list[cluster_id]
            sorted_cluster_tags = sorted(
                cluster_tags, key=lambda x: in_cluster_ranking[x]
            )
            for i, tag in enumerate(sorted_cluster_tags):
                tag_to_position[tag] = i + position

        return tag_to_position

    def sort_tags(
        self, tags: list[str]
    ) -> Tuple[list[str], list[list[str]], list[str]]:
        high_priorities: list[list[str]] = []
        remains: list[str] = tags.copy()

        for i, group in enumerate(self.high_priority_groups):
            high_priorities.append([])

            for tag in remains:
                if tag in group.tags:
                    high_priorities[i].append(tag)
                    remains.remove(tag)
                    continue

        # sort remains with tag_to_position
        on_list: list[str] = []
        not_on_list: list[str] = []
        for tag in remains:
            if tag in self.tag_to_position:
                on_list.append(tag)
            else:
                not_on_list.append(tag)
        on_list.sort(key=lambda x: self.tag_to_position[x])

        return (on_list, high_priorities, not_on_list)


class TagComposer:

    def __init__(
        self,
    ):
        pass
