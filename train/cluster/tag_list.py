from datasets import load_dataset, Dataset

DS_NAME = "isek-ai/danbooru-tags-2024"
REVISION = "202408-at20240906"

OUTPUT_DIR = "./data"


def prepare_dataset():
    ds = load_dataset(DS_NAME, revision=REVISION, split="train")
    assert isinstance(ds, Dataset)
    return ds


def extract_tags(ds: Dataset, column: str):
    tags = set()

    for text in ds[column]:
        if text is None:
            continue

        for tag in text.split(","):
            tags.add(tag.strip())

    return tags


def main():
    ds = prepare_dataset()

    general_tags = extract_tags(ds, "general")
    copyright_tags = extract_tags(ds, "copyright")
    character_tags = extract_tags(ds, "character")
    artist_tags = extract_tags(ds, "artist")
    meta_tags = extract_tags(ds, "meta")

    with open(f"{OUTPUT_DIR}/general_tags.txt", "w") as f:
        f.write("\n".join(sorted(general_tags)))

    with open(f"{OUTPUT_DIR}/copyright_tags.txt", "w") as f:
        f.write("\n".join(sorted(copyright_tags)))

    with open(f"{OUTPUT_DIR}/character_tags.txt", "w") as f:
        f.write("\n".join(sorted(character_tags)))

    with open(f"{OUTPUT_DIR}/artist_tags.txt", "w") as f:
        f.write("\n".join(sorted(artist_tags)))

    with open(f"{OUTPUT_DIR}/meta_tags.txt", "w") as f:
        f.write("\n".join(sorted(meta_tags)))


if __name__ == "__main__":
    main()
