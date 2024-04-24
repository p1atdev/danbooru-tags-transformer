import sys

sys.path.append(".")

from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, AddedToken, Regex
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Split
from tokenizers.normalizers import Lowercase
from transformers import PreTrainedTokenizerFast

from tqdm import tqdm

from src.tags import (
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    SEP_TOKEN,
    MASK_TOKEN,
    GENERAL_START,
    GENERAL_END,
    CHARACTER_START,
    CHARACTER_END,
    COPYRIGHT_START,
    COPYRIGHT_END,
    RATING_SFW,
    RATING_NSFW,
    RATING_GENERAL,
    RATING_SENSITIVE,
    RATING_QUESTIONABLE,
    RATING_EXPLICIT,
    QUALITY_BEST,
    QUALITY_HIGH,
    QUALITY_NORMAL,
    QUALITY_LOW,
    QUALITY_WORST,
    LENGTH_VERY_SHORT,
    LENGTH_SHORT,
    LENGTH_MEDIUM,
    LENGTH_LONG,
    LENGTH_VERY_LONG,
    ASPECT_RATIO_ULTRA_WIDE,
    ASPECT_RATIO_WIDE,
    ASPECT_RATIO_SQUARE,
    ASPECT_RATIO_TALL,
    ASPECT_RATIO_ULTRA_TALL,
    INPUT_START,
    INPUT_END,
    FLAG_KEEP_IDENTITY,
    IDENTITY_LEVEL_NONE,
    IDENTITY_LEVEL_LAX,
    IDENTITY_LEVEL_STRICT,
    RESERVED_TOKENS,
)

DATASET_REPO_ID = "isek-ai/danbooru-tags-2024"
DATASET_REVISION = "202403-at20240423"
DATASET_SPLIT = "train"

PUSH_HUB_NAME = "p1atdev/dart-v2-tokenizer"

SPECIAL_TOKENS = [
    #
    BOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
    SEP_TOKEN,
    MASK_TOKEN,
    #
    GENERAL_START,
    GENERAL_END,
    CHARACTER_START,
    CHARACTER_END,
    COPYRIGHT_START,
    COPYRIGHT_END,
    #
    RATING_SFW,
    RATING_NSFW,
    RATING_GENERAL,
    RATING_SENSITIVE,
    RATING_QUESTIONABLE,
    RATING_EXPLICIT,
    #
    QUALITY_BEST,
    QUALITY_HIGH,
    QUALITY_NORMAL,
    QUALITY_LOW,
    QUALITY_WORST,
    #
    LENGTH_VERY_SHORT,
    LENGTH_SHORT,
    LENGTH_MEDIUM,
    LENGTH_LONG,
    LENGTH_VERY_LONG,
    #
    ASPECT_RATIO_ULTRA_WIDE,
    ASPECT_RATIO_WIDE,
    ASPECT_RATIO_SQUARE,
    ASPECT_RATIO_TALL,
    ASPECT_RATIO_ULTRA_TALL,
    #
    INPUT_START,
    INPUT_END,
    #
    FLAG_KEEP_IDENTITY,
    IDENTITY_LEVEL_NONE,
    IDENTITY_LEVEL_LAX,
    IDENTITY_LEVEL_STRICT,
    #
    *RESERVED_TOKENS,
]
assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))


def prepare_dataset():
    ds = load_dataset(DATASET_REPO_ID, revision=DATASET_REVISION, split=DATASET_SPLIT)
    print(ds)
    return ds


def train_tokenizer(all_tokens: list[str], special_tokens: list[str]):
    tokenizer = Tokenizer(
        # 頭から順番に番号をつける
        WordLevel(
            vocab={tag: i for i, tag in enumerate(all_tokens)}, unk_token=UNK_TOKEN
        )
    )

    tokenizer.normalizer = Lowercase()

    tokenizer.pre_tokenizer = Split(
        pattern=Regex(r",(?:\s)*"), behavior="removed", invert=False
    )

    # register special tokens
    tokenizer.add_special_tokens(
        [
            AddedToken(
                content=token,
            )
            for token in special_tokens
        ]
    )

    # padding
    tokenizer.enable_padding(pad_token=PAD_TOKEN)

    return tokenizer


def convert_to_hf_format(tokenizer_object: Tokenizer):
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object)

    pretrained_tokenizer.bos_token = BOS_TOKEN
    pretrained_tokenizer.eos_token = EOS_TOKEN
    pretrained_tokenizer.pad_token = PAD_TOKEN
    pretrained_tokenizer.unk_token = UNK_TOKEN
    pretrained_tokenizer.sep_token = SEP_TOKEN
    pretrained_tokenizer.mask_token = MASK_TOKEN

    return pretrained_tokenizer


def main():
    ds = prepare_dataset()
    print(ds)

    def get_count_map(column_name: str):
        count_map = {}

        for i, text in tqdm(enumerate(ds[column_name])):
            if text is None:
                continue
            for tag in text.split(", "):
                if tag not in count_map:
                    count_map[tag] = 1
                count_map[tag] += 1

        count_map = {
            k: v for k, v in sorted(count_map.items(), key=lambda x: x[1], reverse=True)
        }
        return count_map

    general_tags_count_map = get_count_map("general")
    copyright_tags_count_map = get_count_map("copyright")
    character_tags_count_map = get_count_map("character")

    popular_general_tags = [
        name for name, count in general_tags_count_map.items() if count > 200
    ]
    popular_copyright_tags = [
        # 登場回数100回未満ののコピーライトタグは、プロンプトに入れても生成されない可能性が高いため、除外する
        name
        for name, count in copyright_tags_count_map.items()
        if count > 100
    ]
    popular_character_tags = [
        # 登場回数100回未満ののキャラクタータグは、プロンプトに入れても生成されない可能性が高いため、除外する
        name
        for name, count in character_tags_count_map.items()
        if count > 100
    ]

    print(f"popular_general_tags: {len(popular_general_tags)}")
    print(f"popular_copyright_tags: {len(popular_copyright_tags)}")
    print(f"popular_character_tags: {len(popular_character_tags)}")

    all_tokens = [
        *SPECIAL_TOKENS,
        *popular_general_tags,
        *popular_copyright_tags,
        *popular_character_tags,
    ]
    assert len(all_tokens) == len(set(all_tokens)), "Duplicate tokens found!"

    tokenizer = train_tokenizer(all_tokens, SPECIAL_TOKENS)

    print(f"tokenizer: {tokenizer}")
    print(f"tokenizer.get_vocab_size(): {tokenizer.get_vocab_size()}")

    test_text = (
        "1girl, 2girls, aaa, long hair, very long hair, "
        "honkai: star rail, arknights, hatsune miku, "
        "hogeeeeeeeee, <|input_end|>, <|pad|>"
    )
    print(f"test_text: {test_text}")
    print(f"tokenizer.encode(test_text): {tokenizer.encode(test_text).tokens}")

    print("Convert to Hugging Face format...")
    pretrained_tokenizer = convert_to_hf_format(tokenizer)
    print(f"pretrained_tokenizer: {pretrained_tokenizer}")

    pretrained_tokenizer.push_to_hub(PUSH_HUB_NAME, private=True)

    print("Done!")


if __name__ == "__main__":
    main()
