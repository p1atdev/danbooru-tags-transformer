from datasets import load_dataset, Dataset
from tqdm import tqdm
import json

JSON_PATH = "data/tag_frequency.json"

DS_NAME = "isek-ai/danbooru-tags-2024"
REVISION = "202408-at20240906"


def prepare_dataset():
    ds = load_dataset(DS_NAME, revision=REVISION, split="train")
    assert isinstance(ds, Dataset)
    return ds


def main():
    tag_freq = {}

    ds = prepare_dataset()

    def extract_tags(text: str | None):
        if text is None:
            return

        tags = text.split(",")
        for tag in tags:
            tag = tag.strip()
            if tag in tag_freq:
                tag_freq[tag] += 1
            else:
                tag_freq[tag] = 1

    all_general_tags = ds["general"]
    all_copyright_tags = ds["copyright"]
    all_character_tags = ds["character"]
    all_meta_tags = ds["meta"]

    for i, _ in tqdm(enumerate(all_general_tags), total=len(all_general_tags)):
        extract_tags(all_general_tags[i])
        extract_tags(all_copyright_tags[i])
        extract_tags(all_character_tags[i])
        extract_tags(all_meta_tags[i])

    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(tag_freq, f, ensure_ascii=False)

    print("Done")


if __name__ == "__main__":
    main()
