import os
import pickle

import torch
import numpy as np

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed

from src.group import TagGroup
from src.cluster import TagCluster
from src.organizer import TagOrganizer
from src.rating import get_rating_tag, get_ambitious_rating_tag
from src.aspect_ratio import calculate_aspect_ratio_tag
from src.length import get_length_tag
from src.tags import (
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


def map_split_tags(examples: Dataset):
    general_list = []
    character_list = []
    copyright_list = []

    for i, id in enumerate(examples["id"]):
        general = examples["general"][i]
        character = examples["character"][i]
        copyright = examples["copyright"][i]

        if character is None:
            character = ""
        if copyright is None:
            copyright = ""

        general_list.append(general)
        character_list.append(character)
        copyright_list.append(copyright)

    return {
        "general": general_list,
        "character": character_list,
        "copyright": copyright_list,
    }


def map_format_tags(examples: Dataset, organizer: TagOrganizer):
    text_list = []

    for i, id in enumerate(examples["id"]):
        general = examples["general"][i]
        character = examples["character"][i]
        copyright = examples["copyright"][i]
        rating = examples["rating"][i]
        image_height = examples["image_height"][i]
        image_width = examples["image_width"][i]

        length_tag = get_length_tag(len(general))
        rating_tag = (
            get_ambitious_rating_tag(rating)
            if np.random.rand() < AMBITIOUS_RATING_RATE
            else get_rating_tag(rating)
        )
        aspect_ratio_tag = calculate_aspect_ratio_tag(image_width, image_height)

        if np.random.rand() < KEEP_IDENTITY_RATE:
            # keep identity

            pre_tags, post_tags = recompose_tags(organizer, general)

            # shuffle pre_tags
            np.random.shuffle(pre_tags)
            # sort post_tags
            post_tags.sort()

            general = pre_tags + [FLAG_KEEP_IDENTITY, INPUT_END] + post_tags

        else:
            result = organizer.organize_tags(general)
            other_tags = sum(result.other_tags, [])  # just flatten

            # randomly split result.other_tags
            np.random.shuffle(other_tags)
            split_index = np.random.randint(0, len(other_tags))
            pre_part = other_tags[:split_index]
            post_part = other_tags[split_index:]

            # drop people tags
            if np.random.rand() < DROP_PEOPLE_RATE:
                post_part.extend(result.people_tags + result.focus_tags)
            else:
                pre_part.extend(result.people_tags + result.focus_tags)

            # shuffle pre_part
            np.random.shuffle(pre_part)
            # sort post_part
            post_part.sort()

            general = pre_part + [INPUT_END] + post_part

        # shuffle copyright and character tags
        np.random.shuffle(copyright)
        np.random.shuffle(character)

        copyright_text = ", ".join(copyright)
        character_text = ", ".join(character)
        general_text = ", ".join(general)

        text = "".join(
            [
                BOS_TOKEN,
                COPYRIGHT_START,
                copyright_text,
                COPYRIGHT_END,
                CHARACTER_START,
                character_text,
                CHARACTER_END,
                rating_tag,
                aspect_ratio_tag,
                length_tag,
                GENERAL_START,
                general_text,
                GENERAL_END,
                EOS_TOKEN,
            ]
        )
        text_list.append(text)

    return {
        "text": text_list,
    }


def map_tokenize_text(example: Dataset, tokenizer: PreTrainedTokenizer):
    input_ids_list = []

    for tag in example["text"]:
        input_ids = tokenizer(tag, padding=False, truncation=False).input_ids
        # remove unk tokens
        input_ids = [i for i in input_ids if i != tokenizer.unk_token_id]

        # shuffle
        np.random.shuffle(input_ids)

        input_ids_list.append(input_ids)

    return {
        "input_ids": input_ids_list,
    }


def main():
    set_seed(SEED)

    tag_cluster = prepare_cluster()
    tag_group = prepare_group()
    tag_organizer = TagOrganizer(tag_group, tag_cluster)

    ds = prepare_dataset()
    tokenizer = prepare_tokenizer()

    # filter out empty text
    ds = ds.filter(
        lambda x: x["text"] is not None and len(x["text"].strip()) > 0,
        batched=False,
        num_proc=NUM_PROC,
    )

    # split tags
    ds = ds.map(
        map_split_tags,
        batched=False,
        num_proc=NUM_PROC,
    )

    # filter too many tags
    ds = ds.filter(lambda x: len(x["general"]) > 100, batched=False, num_proc=NUM_PROC)
    ds = ds.filter(lambda x: len(x["character"]) > 10, batched=False, num_proc=NUM_PROC)
    ds = ds.filter(lambda x: len(x["copyright"]) > 5, batched=False, num_proc=NUM_PROC)

    # tokenize
    ds = ds.map(
        lambda x: tokenize_text(x, tokenizer),
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

    ds.push_to_hub("p1atdev/202402-at20240421-tokenized", max_shard_size="4096MB")


if __name__ == "__main__":
    main()
