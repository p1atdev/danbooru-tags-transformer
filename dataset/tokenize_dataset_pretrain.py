import sys

sys.path.append(".")

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed

from src.composer import TagComposer, TagCluster, TagFrequency

MAX_LENGTH = 256

DATASET_REPO_ID = "isek-ai/danbooru-tags-2024"
REVISION = "202408-at20240906"
DATASET_SPLIT = "train"

TOKENIZER_NAME = "p1atdev/dart-v3-tokenizer-241009"
FREQUENCY_PATH = "data/tag_frequency.json"
CLUSTER_PATH = "data/cluster_map_1024c2.json"

PUSH_ID = "p1atdev/dart-v3-20241009-pretrain-debug"

TEMPERATURE = 1.0
CONDITION_RATE = 0.0

NUM_PROC = 40

SEED = 12345

DEBUG = True


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

    for i, id in enumerate(examples["id"]):
        prompt = composer.compose_pretrain_list(
            general_tags=examples["general"][i],
            copyright_tags=examples["copyright"][i],
            character_tags=examples["character"][i],
            meta_tags=examples["meta"][i],
            rating=examples["rating"][i],
            image_width=examples["image_width"][i],
            image_height=examples["image_height"][i],
            temperature=TEMPERATURE,
            condition_rate=CONDITION_RATE,
        )
        text_list.append(prompt)

    return {
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

    if DEBUG:
        # debug
        ds = ds.select(range(10900))

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
