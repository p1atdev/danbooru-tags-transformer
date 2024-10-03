import sys

sys.path.append(".")


from src.composer import TagComposer, TagCluster, TagFrequency

from transformers import AutoTokenizer
from datasets import load_dataset, Dataset

REPO_ID = "p1atdev/dart-v3-vectors-opt_7-shuffled"

DS_NAME = "isek-ai/danbooru-tags-2024"
REVISION = "202408-at20240906"

TEMPERATURE = 1.0
CONDITION_RATE = 0.0


def map_format(examples, composer: TagComposer):
    prompts = []

    for i, id in enumerate(examples["id"]):
        prompt = composer.compose_pretrain(
            general=examples["general"][i],
            copyright=examples["copyright"][i],
            character=examples["character"][i],
            meta=examples["meta"][i],
            rating=examples["rating"][i],
            image_width=examples["image_width"][i],
            image_height=examples["image_height"][i],
            temperature=TEMPERATURE,
            condition_rate=CONDITION_RATE,
        )
        prompts.append(prompt)

    return {"text": prompts}


def tokenizer_fn(examples, tokenizer):
    return tokenizer(examples["text"])


def main():
    tokenizer = AutoTokenizer.from_pretrained(REPO_ID)

    ds = load_dataset(DS_NAME, revision=REVISION, split="train")
    assert isinstance(ds, Dataset)

    cluster = TagCluster.from_pretrained("data/cluster_map_1024c2.json")
    freq = TagFrequency.from_json("data/tag_frequency.json")
    composer = TagComposer(cluster, freq)

    ds100 = ds.shuffle().select(range(100))

    ds100 = ds100.map(
        map_format,
        batched=True,
        batch_size=1024,
        num_proc=16,
        fn_kwargs={"composer": composer},
        remove_columns=ds100.column_names,
    )

    print(ds100)
    print(ds100[:10])

    ds100 = ds100.filter(lambda x: x["text"] is not None, batched=False)

    print("Filtered")
    print(ds100)

    ds100 = ds100.map(
        tokenizer_fn,
        batched=True,
        batch_size=1024,
        num_proc=16,
        fn_kwargs={"tokenizer": tokenizer},
    )

    print(ds100)
    print(ds100[:10])


if __name__ == "__main__":
    main()
