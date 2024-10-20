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


class PredefinedTags:
    """
    事前に指定したタグの管理をおこなうクラス
    """

    # タグのリスト
    tags: list[str]
    tag_type: PredefinedTagType

    def __init__(self, tags: list[str], tag_type: PredefinedTagType):
        self.tags = tags
        self.tag_type = tag_type

    # テキストファイルから読み込む
    @classmethod
    def from_txt_file(cls, path: str, tag_type: PredefinedTagType) -> "PredefinedTags":
        with open(path, "r") as f:
            tags = f.readlines()
            tags = [tag.strip() for tag in tags if tag.strip()]
        return cls(tags, tag_type)

    @classmethod
    def artistic_error(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/artistic_error.txt", PredefinedTagType.REMOVE)

    @classmethod
    def background(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/background.txt", PredefinedTagType.INSERT_START)

    @classmethod
    def ban_meta(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/ban_meta.txt", PredefinedTagType.BAN)

    @classmethod
    def color_theme(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/color_theme.txt", PredefinedTagType.INSERT_START)

    @classmethod
    def displeasing_meta(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/displeasing_meta.txt", PredefinedTagType.REMOVE)

    @classmethod
    def focus(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/focus.txt", PredefinedTagType.INSERT_START)

    @classmethod
    def people(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/people.txt", PredefinedTagType.INSERT_START)

    @classmethod
    def usable_meta(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/usable_meta.txt", PredefinedTagType.INSERT_START)

    @classmethod
    def medium(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/medium.txt", PredefinedTagType.INSERT_START)

    @classmethod
    def watermark(cls) -> "PredefinedTags":
        return cls.from_txt_file("tags/watermark.txt", PredefinedTagType.REMOVE)


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
        temperature: float = 1.0,
    ) -> Tuple[list[str], list[str]]:
        all_freqs = self.get_frequencies(tags)

        if condition_rate == 0:
            # 条件なしならそのまま返す
            return [], tags

        conditons, others = [], []

        for i, tag in enumerate(tags):
            freqs = avg_softmax(np.array(all_freqs[i:]), temperature)
            criteria = freqs[0] * condition_rate  # 先頭(現在)のタグの確率
            rand = random.random()
            if criteria > rand:
                conditons.append(tag)
            else:
                others.append(tag)
        if len(conditons) == 0 and len(others) == 0:  # 一度も条件が発生しなかった場合
            others = tags

        # 抜けがないかチェック
        assert len(conditons) + len(others) == len(
            tags
        ), f"{len(conditons)} + {len(others)} != {len(tags)}"

        return conditons, others

    # 出現頻度が低いタグを取り除く
    def remove_low_frequency_tags(
        self,
        tags: list[str],
        threshold: int = 100,
    ) -> list[str]:
        return [tag for tag in tags if self.frequency.tag_to_frequency[tag] > threshold]

    # ソートと条件分類を同時におこなう
    def separate_and_sort_tags(
        self,
        tags: list[str],
        condition_rate: float = 0.5,
        temperature: float = 1.0,
    ) -> Tuple[list[list[str]], list[str], list[str]]:
        high_priorities: list[list[str]] = []
        low_priorities: list[str] = self.remove_low_frequency_tags(tags)

        # プライオリティは取り除く
        for i, group in enumerate(self.high_priority_groups):
            high_priorities.append([])

            for tag in (
                low_priorities.copy()
            ):  # must copy to remove elements later in the loop
                if tag in group.tags:
                    high_priorities[i].append(tag)
                    low_priorities.remove(tag)

        # 条件付にぶち込むタグと、生成する側のタグに分ける
        conditions, remains = [], []

        # クラスターごとに分類
        cluster_tags = self.clustering_tags(low_priorities)
        for cluster_id, in_cluster_tags in cluster_tags.items():
            # 条件に入れるタグと入れないタグを分類
            conditons, others = self.random_conditioning(
                in_cluster_tags, condition_rate, temperature
            )
            conditions.extend(conditons)
            remains.extend(others)

        # remains はソートする
        remains = sorted(remains, key=lambda x: self.tag_to_position[x])

        return (high_priorities, conditions, remains)

    # 単純に出現頻度順にソートする
    def sort_tags_by_frequency(
        self,
        tags: list[str],
    ) -> list[str]:
        return sorted(tags, key=lambda x: self.frequency.tag_to_frequency[x])


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
    ]

    def __init__(self, cluster: TagCluster, frequency: TagFrequency):
        self.cluster = cluster
        self.frequency = frequency

        predefined_tags = self.predefined_general_tags

        self.selector = TagSelector(cluster, frequency, predefined_tags)

    def compose_pretrain_list(
        self,
        general_tags: list[str],
        copyright_tags: list[str],
        character_tags: list[str],
        meta_tags: list[str],
        rating: SHORT_RATING_TAG,
        image_width: int,
        image_height: int,
        temperature: float = 1.0,
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

        # タグをソート
        high_priorities, _conditions, low_priorities = (
            self.selector.separate_and_sort_tags(
                general_tags,
                condition_rate=condition_rate,
                temperature=temperature,
            )
        )
        top_insert_tags = []
        for tags, predefined in zip(
            high_priorities, self.selector.high_priority_groups, strict=True
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                top_insert_tags.extend(self.selector.sort_tags_by_frequency(tags))

        assert isinstance(meta_tags, list)
        ok_meta_tags = []
        for predefined in self.predefined_meta_tags:
            for tag_part in predefined.tags:
                for tag in meta_tags.copy():  # 部分的にでも含まれていたら
                    if tag_part in tag:
                        if predefined.tag_type == PredefinedTagType.BAN:
                            # BAN row
                            return None
                        elif predefined.tag_type == PredefinedTagType.REMOVE:
                            # just remove
                            meta_tags.remove(tag)
                            continue
                        elif predefined.tag_type == PredefinedTagType.INSERT_START:
                            # do nothing
                            ok_meta_tags.append(tag)
        meta_tags = self.selector.sort_tags_by_frequency(ok_meta_tags)

        # 出現頻度順にソート
        character_tags = self.selector.sort_tags_by_frequency(character_tags)
        copyright_tags = self.selector.sort_tags_by_frequency(copyright_tags)

        # テンプレートに適用
        prompt = format_completion(
            priority=top_insert_tags,
            general=low_priorities,
            character=character_tags,
            copyright=copyright_tags,
            meta=meta_tags,
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
        temperature: float = 1.0,
        condition_rate: float = 0.0,
        full_dropout_rate: float = 0.05,  # 5%の確率で全てのgeneralタグをドロップして条件に含めない
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

        # タグをソート
        high_priorities, conditions, low_priorities = (
            self.selector.separate_and_sort_tags(
                general_tags,
                condition_rate=condition_rate,
                temperature=temperature,
            )
        )
        if is_full_dropout:
            # 条件部分を低優先度に移動
            low_priorities.extend(conditions)
            conditions = []
            low_priorities = self.selector.sort_tags_by_frequency(low_priorities)

        top_insert_tags = []
        for tags, predefined in zip(
            high_priorities, self.selector.high_priority_groups, strict=True
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                top_insert_tags.extend(tags)

        assert isinstance(meta_tags, list)
        ok_meta_tags = []
        for predefined in self.predefined_meta_tags:
            for tag_part in predefined.tags:
                for tag in meta_tags.copy():  # 部分的にでも含まれていたら
                    if tag_part in tag:
                        if predefined.tag_type == PredefinedTagType.BAN:
                            # BAN row
                            return None
                        elif predefined.tag_type == PredefinedTagType.REMOVE:
                            # just remove
                            meta_tags.remove(tag)
                            continue
                        elif predefined.tag_type == PredefinedTagType.INSERT_START:
                            # do nothing
                            ok_meta_tags.append(tag)
        meta_tags = self.selector.sort_tags_by_frequency(ok_meta_tags)

        # 条件部分
        if is_full_dropout:
            # 条件部分を全部補完側に移動
            meta_condition_tags: list[str] = []
            meta_remains_tags = meta_tags
            condition_tags = []
        else:
            meta_condition_tags, meta_remains_tags = random_choose(
                meta_tags, condition_rate
            )  # condition_rateの確率でmeta_tagsを含める
            condition_tags = top_insert_tags + conditions + meta_condition_tags

        # オリジナルなら original タグを確率でドロップ
        if copyright_tags == ["original"] and character_tags == []:
            if random.random() < 0.5:  # 50%の確率でドロップ
                copyright_tags = []

        # シャッフル
        random.shuffle(condition_tags)
        random.shuffle(character_tags)
        random.shuffle(copyright_tags)

        # まとめてシャッフル
        rating_aspect_ratio_length = [rating_tag, aspect_ratio_tag, length_tag]
        random.shuffle(rating_aspect_ratio_length)

        # 生成部分
        if is_full_dropout:
            meta_general = top_insert_tags + meta_remains_tags + low_priorities
        else:
            meta_general = meta_remains_tags + low_priorities

        # テンプレートに適用
        prompt = format_sft_with_initial_condition(
            rating_aspect_ratio_length=rating_aspect_ratio_length,  # shuffled
            condition=condition_tags,  # shuffled
            copyright=copyright_tags,  # shuffled
            character=character_tags,  # shuffled
            meta_general=meta_general,
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
        temperature: float = 1.0,
        condition_rate: float = 0.0,
        full_dropout_rate: float = 0.05,  # 5%の確率で全てのgeneralタグをドロップして条件に含めない
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

        # タグをソート
        high_priorities, conditions, low_priorities = (
            self.selector.separate_and_sort_tags(
                general_tags,
                condition_rate=condition_rate,
                temperature=temperature,
            )
        )
        # use の時は条件も生成部分に追加する
        low_priorities.extend(conditions)
        low_priorities = self.selector.sort_tags_by_frequency(low_priorities)
        if is_full_dropout:
            # 条件なし
            conditions = []

        top_insert_tags = []
        for tags, predefined in zip(
            high_priorities, self.selector.high_priority_groups, strict=True
        ):
            if predefined.tag_type == PredefinedTagType.BAN:
                if len(tags) > 0:
                    # BAN row
                    return None
            elif predefined.tag_type == PredefinedTagType.REMOVE:
                # just remove
                continue
            elif predefined.tag_type == PredefinedTagType.INSERT_START:
                top_insert_tags.extend(tags)

        assert isinstance(meta_tags, list)
        ok_meta_tags = []
        for predefined in self.predefined_meta_tags:
            for tag_part in predefined.tags:
                for tag in meta_tags.copy():  # 部分的にでも含まれていたら
                    if tag_part in tag:
                        if predefined.tag_type == PredefinedTagType.BAN:
                            # BAN row
                            return None
                        elif predefined.tag_type == PredefinedTagType.REMOVE:
                            # just remove
                            meta_tags.remove(tag)
                            continue
                        elif predefined.tag_type == PredefinedTagType.INSERT_START:
                            # do nothing
                            ok_meta_tags.append(tag)
        meta_tags = self.selector.sort_tags_by_frequency(ok_meta_tags)

        # 条件部分
        if is_full_dropout:
            # 条件部分を全部補完側に移動
            meta_condition_tags: list[str] = []
            meta_remains_tags = meta_tags
            condition_tags = []
        else:
            meta_condition_tags, meta_remains_tags = random_choose(
                meta_tags, condition_rate
            )  # condition_rateの確率でmeta_tagsを含める
            condition_tags = top_insert_tags + conditions + meta_condition_tags

        # オリジナルなら original タグを確率でドロップ
        if copyright_tags == ["original"] and character_tags == []:
            if random.random() < 0.5:  # 50%の確率でドロップ
                copyright_tags = []

        # シャッフル
        random.shuffle(condition_tags)
        random.shuffle(character_tags)
        random.shuffle(copyright_tags)

        # まとめてシャッフル
        rating_aspect_ratio_length = [rating_tag, aspect_ratio_tag, length_tag]
        random.shuffle(rating_aspect_ratio_length)

        # 生成部分
        meta_general = top_insert_tags + meta_remains_tags + low_priorities

        # テンプレートに適用
        prompt = format_sft_with_use_condition(
            rating_aspect_ratio_length=rating_aspect_ratio_length,  # shuffled
            condition=condition_tags,  # shuffled
            copyright=copyright_tags,  # shuffled
            character=character_tags,  # shuffled
            meta_general=meta_general,
        )

        return prompt
