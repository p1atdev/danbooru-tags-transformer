import sys

sys.path.append(".")

import random
import numpy as np

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed

from src.composer import TagComposer, TagCluster, TagFrequency
from models.ndart.processing_ndart import NDartProcessor


MAX_LENGTH = 256

DATASET_REPO_ID = "isek-ai/danbooru-tags-2024"
REVISION = "202408-at20240906"
DATASET_SPLIT = "train"

ENCODER_TOKENIZER_NAME = "intfloat/multilingual-e5-base"
DECODER_TOKENIZER_NAME = "p1atdev/dart-v3-tokenizer-241010"
NATURAL_PLACEHOLDER = "<|natural|>"

FREQUENCY_PATH = "data/tag_frequency.json"
CLUSTER_PATH = "data/general_1024cluster_opt17.json"

PUSH_ID = "p1atdev/dart-v3-20241102-ndart-1"

YEAR_MIN = 2017

NUM_PROC = 40

SEED = 12345

DEBUG = True

# FULL_DROPOUT_RATE = 0.05  # general タグを全部条件から外す確率


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
    rand = power_distribution_random(n=2, size=batch_size)
    return rand.tolist()


def prepare_dataset():
    ds = load_dataset(DATASET_REPO_ID, revision=REVISION, split=DATASET_SPLIT)
    assert isinstance(ds, Dataset)

    return ds


def prepare_processor():
    processor = NDartProcessor(
        encoder_tokenizer=AutoTokenizer.from_pretrained(ENCODER_TOKENIZER_NAME),
        decoder_tokenizer=AutoTokenizer.from_pretrained(DECODER_TOKENIZER_NAME),
        natural_token=NATURAL_PLACEHOLDER,
    )

    return processor


def filter_by_year(examples: Dataset):
    flags = []
    for date in examples["created_at"]:
        year = int(date.split("-")[0])
        flags.append(year >= YEAR_MIN)

    return flags


def filter_by_score(examples: Dataset):
    flags = []
    for i, score in enumerate(examples["score"]):
        rating = examples["rating"][i]
        if rating == "g":
            flags.append(score >= 0)
        elif rating == "s":
            flags.append(score >= 1)
        elif rating == "q":
            flags.append(score >= 3)
        elif rating == "e":
            flags.append(score >= 3)

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
    tag_list = []
    natural_list = []

    batch_size = len(examples["id"])
    # ランダムに確率を変動させる
    condition_rates = get_condition_rate(batch_size)

    for i, condition_rate in enumerate(condition_rates):
        prompt = composer.compose_ndart_list(
            general_tags=examples["general"][i],
            copyright_tags=examples["copyright"][i],
            character_tags=examples["character"][i],
            meta_tags=examples["meta"][i],
            rating=examples["rating"][i],
            image_width=examples["image_width"][i],
            image_height=examples["image_height"][i],
            condition_rate=condition_rate,
        )
        tag_list.append(prompt)

        natural_prompt = composer.compose_natural_list(
            general_tags=examples["general"][i],
            character_tags=examples["character"][i],
            copyright_tags=examples["copyright"][i],
        )
        natural_list.append(natural_prompt)

    return {
        "id": examples["id"],
        "tag": tag_list,
        "natural": natural_list,  # とりま同じものを試してみる
    }


def map_tokenize_text(example: Dataset, processor):
    inputs = processor(tag_text=example["tag"], natural_text=example["natural"])
    return inputs


def main():
    set_seed(SEED)

    cluster = TagCluster.from_pretrained(CLUSTER_PATH)
    freq = TagFrequency.from_json(FREQUENCY_PATH)

    tag_composer = TagComposer(
        cluster=cluster,
        frequency=freq,
    )

    ds = prepare_dataset()
    processor = prepare_processor()

    # filter by year
    ds = ds.filter(
        filter_by_year,
        batched=True,
        batch_size=1024,
        num_proc=NUM_PROC,
    )

    # filter by score
    ds = ds.filter(
        filter_by_score,
        batched=True,
        batch_size=1024,
        num_proc=NUM_PROC,
    )

    #! filter only original
    ds = ds.filter(
        lambda x: (x["copyright"] == "original" or x["copyright"] is None)
        and x["character"] is None,
        batched=False,
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
        lambda x: processor.decoder_tokenizer.unk_token_id
        not in processor.decoder_tokenizer.encode_plus(
            x["copyright"], add_special_tokens=False
        ).input_ids,
        batched=False,
        num_proc=NUM_PROC,
    )
    ds = ds.filter(
        lambda x: processor.decoder_tokenizer.unk_token_id
        not in processor.decoder_tokenizer.encode_plus(
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
        fn_kwargs={"tokenizer": processor.decoder_tokenizer},
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
        load_from_cache_file=False,
    )

    # filter None
    ds = ds.filter(
        lambda x: x["tag"] is not None,
        batched=False,
        num_proc=NUM_PROC,
        load_from_cache_file=False,
    )

    # # tokenize
    ds = ds.map(
        map_tokenize_text,
        batched=True,
        num_proc=NUM_PROC,
        fn_kwargs={"processor": processor},
        load_from_cache_file=False,
    )

    # train test split
    ds = ds.train_test_split(
        test_size=10000 if not DEBUG else 10,
        shuffle=True,
        load_from_cache_file=False,
    )

    ds.push_to_hub(
        PUSH_ID,
        max_shard_size="4096MB",
        private=True,
    )


if __name__ == "__main__":
    main()
