import sys

sys.path.append(".")

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"  # to prevent OpenBLAS's error

import numpy as np

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed

from src.group import TagGroup
from src.cluster import TagCluster
from src.organizer import GroupTagOrganizer, ClusterTagOrganizer
from src.composer import TagComposer
from src.tags import IDENTITY_LEVEL_NONE, IDENTITY_LEVEL_LAX, IDENTITY_LEVEL_STRICT
from src.formatter import format_sft

from tokenize_dataset_pretrain import map_tokenize_text, map_split_tags

MAX_LENGTH = 256

PUSH_HUB_NAME = "p1atdev/dart-v2-20240426-sft"

DATASET_REPO_ID = "isek-ai/danbooru-tags-2024"
REVISION = "202403-at20240422"
DATASET_SPLIT = "train"

TOKENIZER_NAME = "p1atdev/dart-v2-tokenizer"

FUZZY_RATING_RATE = 0.25
DROP_PEOPLE_RATE = 0.1
CONDITION_RATE = 0.5
COPYRIGHT_CHARACTER_AUGMENTATION_RATE = 1.25
IDENTITY_LEVEL_RATES = {
    "lax": 0.25,
    "strict": 0.25,
    "none": 0.5,
}

NUM_PROC = 1

SEED = 12345

DEBUG = True


def prepare_group():
    group = TagGroup()

    return group


def prepare_cluster(
    embedding_model_name: str,
    num_clusters: int,
    num_init: int,
    max_iter: int,
    save_path: str,
):
    if os.path.exists(save_path):
        print("Loading cluster...")
        cluster = TagCluster.from_pretrained(save_path)
        print("Cluster loaded.")
    else:
        print("Training cluster...")
        cluster = TagCluster.train_from_embedding_model(
            embedding_model_name=embedding_model_name,
            n_clusters=num_clusters,
            n_init=num_init,
            max_iter=max_iter,
        )
        cluster.save_pretrained(save_path)
        print("Cluster trained and saved.")

    return cluster


DEFAULT_TAG_GROUP = prepare_group()
EMBEDDING_CLUSTERS = {
    "lax": prepare_cluster(
        embedding_model_name="p1atdev/dart-v2-vectors",
        num_clusters=512,
        num_init=20,
        max_iter=500,
        save_path="data/cluster_map_512c.json",
    ),
    "strict": prepare_cluster(
        embedding_model_name="p1atdev/dart-v2-vectors",
        num_clusters=256,  # fewer than lax
        num_init=20,
        max_iter=500,
        save_path="data/cluster_map_256c.json",
    ),
}
TAG_ORGANIZERS = {
    "none": GroupTagOrganizer(DEFAULT_TAG_GROUP),
    "lax": ClusterTagOrganizer(DEFAULT_TAG_GROUP, EMBEDDING_CLUSTERS["lax"]),
    "strict": ClusterTagOrganizer(DEFAULT_TAG_GROUP, EMBEDDING_CLUSTERS["strict"]),
}
IDENTITY_LEVEL_TAGS = {
    "none": IDENTITY_LEVEL_NONE,
    "lax": IDENTITY_LEVEL_LAX,
    "strict": IDENTITY_LEVEL_STRICT,
}


def prepare_dataset():
    ds = load_dataset(DATASET_REPO_ID, revision=REVISION, split=DATASET_SPLIT)
    assert isinstance(ds, Dataset)

    return ds


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    return tokenizer


def map_format_tags(examples: Dataset, composer: TagComposer):
    text_list = []

    for i, id in enumerate(examples["id"]):
        general = examples["general"][i]
        character = examples["character"][i]
        copyright = examples["copyright"][i]
        rating = examples["rating"][i]
        image_width = examples["image_width"][i]
        image_height = examples["image_height"][i]

        if len(character) > 0 or len(copyright) > 0:
            # 版権タグがある場合はidentity_levelをstrictにする
            identity_level = "strict"
        else:
            # 処理分岐でidentity_levelを変える
            identity_level: str = np.random.choice(
                list(IDENTITY_LEVEL_RATES.keys()), p=list(IDENTITY_LEVEL_RATES.values())
            )

        # parse and organize tags
        result = TAG_ORGANIZERS[identity_level].organize_tags(general)
        components = composer.get_components_identity_keep(
            rating=rating,
            copyright=copyright,
            character=character,
            organizer_result=result,
            image_width=image_width,
            image_height=image_height,
        )

        # format a prompt
        prompt = format_sft(components, IDENTITY_LEVEL_TAGS[identity_level])

        text_list.append(prompt)

    return {
        "text": text_list,
    }


def main():
    set_seed(SEED)

    tag_composer = TagComposer(
        fuzzy_rating_tag_rate=FUZZY_RATING_RATE,
        drop_people_rate=DROP_PEOPLE_RATE,
        condition_rate=CONDITION_RATE,
        copyright_character_augmentation_rate=COPYRIGHT_CHARACTER_AUGMENTATION_RATE,
    )

    ds = prepare_dataset()
    tokenizer = prepare_tokenizer()

    if DEBUG:
        ds = ds.select(range(1000))

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
        lambda x: map_split_tags(x, tokenizer),
        batched=True,
        num_proc=NUM_PROC,
    )

    # filter too many tags
    ds = ds.filter(lambda x: len(x["general"]) <= 100, batched=False, num_proc=NUM_PROC)
    ds = ds.filter(
        lambda x: len(x["character"]) <= 10, batched=False, num_proc=NUM_PROC
    )
    ds = ds.filter(lambda x: len(x["copyright"]) <= 5, batched=False, num_proc=NUM_PROC)

    # format tags
    ds = ds.map(
        lambda x: map_format_tags(x, tag_composer),
        batched=True,
        num_proc=NUM_PROC,
    )

    # tokenize
    ds = ds.map(
        lambda x: map_tokenize_text(x, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
        num_proc=NUM_PROC,
    )

    # train test split
    ds = ds.train_test_split(
        test_size=10000 if not DEBUG else 10,
    )

    ds.push_to_hub(
        PUSH_HUB_NAME,
        max_shard_size="4096MB",
        private=True,
    )


if __name__ == "__main__":
    main()
