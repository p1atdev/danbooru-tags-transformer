import sys

sys.path.append(".")

import random
import numpy as np

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed

from src.composer import TagComposer, TagCluster, TagFrequency

MAX_LENGTH = 256

DATASET_REPO_ID = "isek-ai/danbooru-tags-2024"
REVISION = "202408-at20240906"
DATASET_SPLIT = "train"

TOKENIZER_NAME = "p1atdev/dart-v3-tokenizer-240912"
FREQUENCY_PATH = "data/tag_frequency.json"
CLUSTER_PATH = "data/cluster_map_1024c2.json"

PUSH_ID = "p1atdev/dart-v3-20241006-sft"

YEAR_MIN = 2018

NUM_PROC = 40

SEED = 12345

DEBUG = False


# べき乗を使ったランダムな値を生成する関数
def power_distribution_random(
    n: float = 1.25,
    size: int = 1,
):
    """
    0から1の範囲でべき乗に基づいたランダムな値を生成する関数。

    Parameters:
    - n: べき乗の指数 (n > 1 なら 0 寄り、n < 1 なら 1 寄り)
    - size: 生成する乱数の個数

    Returns:
    - np.array: ランダムな値のリスト
    """
    return np.array([random.random() ** n for _ in range(size)])


# ガウス分布（正規分布）を使ったランダムな値を生成する関数
def gaussian_distribution_random(
    mean: float = 1.0,
    std_dev: float = 0.1,
    size: int = 1,
    min_value: float = 0.0,
    max_value: float = 2.0,
):
    """
    ガウス分布（正規分布）に基づいたランダムな値を生成し、0から2の範囲にクリップする関数。

    Parameters:
    - mean: 平均
    - std_dev: 標準偏差
    - size: 生成する乱数の個数

    Returns:
    - np.array: ランダムな値のリスト
    """
    return np.clip(np.random.normal(mean, std_dev, size), min_value, max_value)


# 条件とする確率を生成 (0.0 から 1.0 の範囲で0寄り多め)
def get_condition_rate(batch_size: int = 1) -> list[float]:
    rand = power_distribution_random(size=batch_size)
    return rand.tolist()


# temperature を生成 (0~2で1寄り)
def get_temperature(batch_size: int = 1) -> list[float]:
    rand = gaussian_distribution_random(size=batch_size)
    return rand.tolist()


def prepare_dataset():
    ds = load_dataset(DATASET_REPO_ID, revision=REVISION, split=DATASET_SPLIT)
    assert isinstance(ds, Dataset)

    return ds


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    return tokenizer


def filter_by_year(examples: Dataset):
    flags = []
    for date in examples["created_at"]:
        year = int(date.split("-")[0])
        flags.append(year >= YEAR_MIN)

    return flags


def map_split_tags(examples: Dataset, tokenizer: PreTrainedTokenizer):
    general_list = []
    character_list = []
    copyright_list = []
    meta_list = []

    for i, id in enumerate(examples["id"]):
        general: str = examples["general"][i]
        character: str = examples["character"][i]
        copyright: str = examples["copyright"][i]
        meta: str = examples["meta"][i]

        if character is None:
            character_tags = []
        else:
            character_tags = [
                tag.strip() for tag in character.split(", ") if tag.strip() != ""
            ]
        if copyright is None:
            copyright_tags = []
        else:
            copyright_tags = [
                tag.strip() for tag in copyright.split(", ") if tag.strip() != ""
            ]
        if meta is None:
            meta_tags = []
        else:
            meta_tags = [tag.strip() for tag in meta.split(", ") if tag.strip() != ""]

        assert isinstance(character_tags, list)
        assert isinstance(copyright_tags, list)
        assert isinstance(meta_tags, list)

        character_list.append(character_tags)
        copyright_list.append(copyright_tags)
        meta_list.append(meta_tags)

        # encode general tags and remove unk tokens, then decode
        general_token_ids = tokenizer.encode_plus(
            general, add_special_tokens=False
        ).input_ids
        general_token_ids = [
            token_id
            for token_id in general_token_ids
            if token_id != tokenizer.unk_token_id
        ]
        general_tags = tokenizer.batch_decode(general_token_ids)
        general_list.append(general_tags)

    return {
        "general": general_list,
        "character": character_list,
        "copyright": copyright_list,
        "meta": meta_list,
    }


def map_format_tags(examples: Dataset, composer: TagComposer):
    text_list = []

    batch_size = len(examples["id"])
    # ランダムに確率を変動させる
    condition_rates = get_condition_rate(batch_size)
    temperatures = get_temperature(batch_size)

    for i, (condition_rate, temperature) in enumerate(
        zip(condition_rates, temperatures, strict=True)
    ):
        prompt = composer.compose_sft_list(
            general_tags=examples["general"][i],
            copyright_tags=examples["copyright"][i],
            character_tags=examples["character"][i],
            meta_tags=examples["meta"][i],
            rating=examples["rating"][i],
            image_width=examples["image_width"][i],
            image_height=examples["image_height"][i],
            temperature=temperature,
            condition_rate=condition_rate,
        )
        text_list.append(prompt)

    return {
        "id": examples["id"],
        "text": text_list,
    }


def map_tokenize_text(example: Dataset, tokenizer: PreTrainedTokenizer):
    tokenized = tokenizer(example["text"])
    input_ids = tokenized["input_ids"]

    return {
        "input_ids": input_ids,
    }


def main():
    set_seed(SEED)

    cluster = TagCluster.from_pretrained(CLUSTER_PATH)
    freq = TagFrequency.from_json(FREQUENCY_PATH)

    tag_composer = TagComposer(
        cluster=cluster,
        frequency=freq,
    )

    ds = prepare_dataset()
    tokenizer = prepare_tokenizer()

    # filter by year
    ds = ds.filter(
        filter_by_year,
        batched=True,
        batch_size=1024,
        num_proc=NUM_PROC,
    )

    if DEBUG:
        # debug
        ds = ds.select(range(10000))

    # filter out empty text
    ds = ds.filter(
        lambda x: x["general"] is not None and len(x["general"].strip()) > 0,
        batched=False,
        num_proc=NUM_PROC,
    )

    # replace null with empty text in copyright and character
    ds = ds.map(
        lambda x: {
            "copyright": x["copyright"] if x["copyright"] is not None else "",
            "character": x["character"] if x["character"] is not None else "",
        },
        batched=False,
        num_proc=NUM_PROC,
    )

    # filter out if copyright or character is unknown
    ds = ds.filter(
        lambda x: tokenizer.unk_token_id
        not in tokenizer.encode_plus(
            x["copyright"], add_special_tokens=False
        ).input_ids,
        batched=False,
        num_proc=NUM_PROC,
    )
    ds = ds.filter(
        lambda x: tokenizer.unk_token_id
        not in tokenizer.encode_plus(
            x["character"], add_special_tokens=False
        ).input_ids,
        batched=False,
        num_proc=NUM_PROC,
    )

    # split tags
    ds = ds.map(
        map_split_tags,
        batched=True,
        num_proc=NUM_PROC,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # filter too many tags
    ds = ds.filter(lambda x: len(x["general"]) <= 128, batched=False, num_proc=NUM_PROC)
    ds = ds.filter(
        lambda x: len(x["character"]) <= 10, batched=False, num_proc=NUM_PROC
    )
    ds = ds.filter(lambda x: len(x["copyright"]) <= 5, batched=False, num_proc=NUM_PROC)

    # format tags
    ds = ds.map(
        map_format_tags,
        batched=True,
        num_proc=NUM_PROC,
        fn_kwargs={"composer": tag_composer},
        remove_columns=ds.column_names,
    )

    # filter None
    ds = ds.filter(
        lambda x: x["text"] is not None,
        batched=False,
        num_proc=NUM_PROC,
    )

    # tokenize
    ds = ds.map(
        map_tokenize_text,
        batched=True,
        num_proc=NUM_PROC,
        fn_kwargs={"tokenizer": tokenizer},
    )

    # train test split
    ds = ds.train_test_split(
        test_size=10000 if not DEBUG else 10,
        shuffle=True,
    )

    ds.push_to_hub(
        PUSH_ID,
        max_shard_size="4096MB",
        private=True,
    )


if __name__ == "__main__":
    main()
