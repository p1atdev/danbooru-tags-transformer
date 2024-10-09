import torch
import numpy as np

from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, set_seed

MAX_LENGTH = 256

DATASET_REPO_ID = "isek-ai/danbooru-tags-2024"
REVISION = "202408-at20240906"
DATASET_SPLIT = "train"

TOKENIZER_NAME = "p1atdev/dart-v3-tokenizer-241009"

NUM_PROC = 40

SEED = 12345

DO_SHUFFLE = True
PUSH_REPO_ID = "p1atdev/202408-at20240906-tokenized-shuffle-241009"


def prepare_dataset():
    ds = load_dataset(DATASET_REPO_ID, revision=REVISION, split=DATASET_SPLIT)
    assert isinstance(ds, Dataset)

    return ds


def prepare_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)

    return tokenizer


def tokenize_text(example: Dataset, tokenizer: PreTrainedTokenizer):
    input_ids_list = []

    for tag in example["text"]:
        input_ids = tokenizer(tag, padding=False, truncation=False).input_ids
        # remove unk tokens
        input_ids = [i for i in input_ids if i != tokenizer.unk_token_id]

        if DO_SHUFFLE:
            # shuffle
            np.random.shuffle(input_ids)

        input_ids_list.append(input_ids)

    return {
        "input_ids": input_ids_list,
    }


def main():
    set_seed(SEED)

    ds = prepare_dataset()
    tokenizer = prepare_tokenizer()

    # rename column "general" to "text"
    ds = ds.rename_column("general", "text")
    other_column_names = [col for col in ds.column_names if col != "text"]
    ds = ds.remove_columns(other_column_names)

    # filter out empty text
    ds = ds.filter(
        lambda x: x["text"] is not None and len(x["text"].strip()) > 0,
        batched=False,
        num_proc=NUM_PROC,
    )

    # tokenize
    ds = ds.map(
        lambda x: tokenize_text(x, tokenizer),
        batched=True,
        remove_columns=ds.column_names,
        num_proc=NUM_PROC,
    )

    # filter out short input_ids
    ds = ds.filter(
        lambda x: len(x["input_ids"]) > 10,
        batched=False,
        num_proc=NUM_PROC,
    )

    # train test split
    ds = ds.train_test_split(
        test_size=10000,
    )

    ds.push_to_hub(PUSH_REPO_ID, max_shard_size="4096MB")


if __name__ == "__main__":
    main()
