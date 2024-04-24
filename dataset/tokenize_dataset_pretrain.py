import os

import torch
import numpy as np

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed

from src.group import TagGroup
from src.cluster import TagCluster
from src.organizer import TagOrganizer
from src.composer import TagComposer
from src.tags import FLAG_KEEP_IDENTITY

MAX_LENGTH = 256

DATASET_REPO_ID = "isek-ai/danbooru-tags-2024"
REVISION = "202403-at20240422"
DATASET_SPLIT = "train"

TOKENIZER_NAME = "p1atdev/dart-tokenizer-v2-encode"

FUZZY_RATING_RATE = 0.25
DROP_PEOPLE_RATE = 0.1
KEEP_IDENTITY_RATE = 0.5
KEEP_IDENTITY_CONDITION_RATE = 0.5

EMBEDDING_MODEL_NAME = "p1atdev/dart2vec-opt_6"
NUM_CLUSTERS = 40
NUM_INIT = 10
MAX_ITER = 1000
CLUSTER_MAP_PATH = "data/cluster_map.json"

NUM_PROC = 40

SEED = 12345


def prepare_cluster():
    if os.path.exists(CLUSTER_MAP_PATH):
        cluster = TagCluster.from_pretrained(CLUSTER_MAP_PATH)
    else:
        cluster = TagCluster.train_from_embedding_model(
            EMBEDDING_MODEL_NAME,
            n_clusters=NUM_CLUSTERS,
            n_init=NUM_INIT,
            max_iter=MAX_ITER,
        )
        cluster.save_pretrained(CLUSTER_MAP_PATH)

    return cluster


def prepare_group():
    group = TagGroup()

    return group


def prepare_dataset():
    ds = load_dataset(DATASET_REPO_ID, revision=REVISION, split=DATASET_SPLIT)
    assert isinstance(ds, Dataset)

    return ds


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    return tokenizer


def map_split_tags(examples: Dataset, tokenizer: PreTrainedTokenizer):
    general_list = []
    character_list = []
    copyright_list = []

    for i, id in enumerate(examples["id"]):
        general: str = examples["general"][i]
        character: str = examples["character"][i]
        copyright: str = examples["copyright"][i]

        if character is None:
            character = ""
        if copyright is None:
            copyright = ""

        character_list.append((character).split(", "))
        copyright_list.append((copyright).split(", "))

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
    }


def map_format_tags(examples: Dataset, composer: TagComposer):
    text_list = []

    for i, id in enumerate(examples["id"]):
        general = examples["general"][i]
        character = examples["character"][i]
        copyright = examples["copyright"][i]
        rating = examples["rating"][i]
        image_width = examples["image_width"][i]
        image_height = examples["image_height"][i]

        prompt = composer.compose_prompt(
            rating=rating,
            copyright=copyright,
            character=character,
            general=general,
            image_width=image_width,
            image_height=image_height,
        )

        text_list.append(prompt)

    return {
        "text": text_list,
    }


def map_tokenize_text(example: Dataset, tokenizer: PreTrainedTokenizer):
    input_ids_list = []

    for tag in example["text"]:
        input_ids = tokenizer(tag, padding=False, truncation=False).input_ids

        input_ids_list.append(input_ids)

    return {
        "input_ids": input_ids_list,
    }


def main():
    set_seed(SEED)

    tag_cluster = prepare_cluster()
    tag_group = prepare_group()
    tag_organizer = TagOrganizer(tag_group, tag_cluster)
    tag_composer = TagComposer(
        tag_organizer,
        keep_identity_token=FLAG_KEEP_IDENTITY,
    )

    ds = prepare_dataset()
    tokenizer = prepare_tokenizer()

    # filter out empty text
    ds = ds.filter(
        lambda x: x["text"] is not None and len(x["text"].strip()) > 0,
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
        lambda x: map_split_tags(x, tokenizer),
        batched=False,
        num_proc=NUM_PROC,
    )

    # filter too many tags
    ds = ds.filter(lambda x: len(x["general"]) > 100, batched=False, num_proc=NUM_PROC)
    ds = ds.filter(lambda x: len(x["character"]) > 10, batched=False, num_proc=NUM_PROC)
    ds = ds.filter(lambda x: len(x["copyright"]) > 5, batched=False, num_proc=NUM_PROC)

    # format tags
    ds = ds.map(
        lambda x: map_format_tags(x, tag_composer),
        batched=False,
        num_proc=NUM_PROC,
    )

    # tokenize
    ds = ds.map(
        lambda x: map_tokenize_text(x, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
        num_proc=NUM_PROC,
    )

    # filter out short input_ids
    ds = ds.filter(lambda x: len(x["input_ids"]) > 10, batched=False)

    # train test split
    ds = ds.train_test_split(
        test_size=10000,
    )

    ds.push_to_hub("p1atdev", max_shard_size="4096MB", private=True)


if __name__ == "__main__":
    main()
