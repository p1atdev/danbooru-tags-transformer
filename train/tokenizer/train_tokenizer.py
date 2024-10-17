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
    CommonSpecialTokens,
    TagCategoryTokens,
    RatingTokens,
    QualityTokens,
    LengthTokens,
    AspectRatioTokens,
    InstructionTokens,
    IdentityTokens,
    MultiModalTokens,
    TranslationTokens,
    RESERVED_TOKENS,
)

DATASET_REPO_ID = "isek-ai/danbooru-tags-2024"
DATASET_REVISION = "202408-at20240906"
DATASET_SPLIT = "train"

PUSH_HUB_NAME = "p1atdev/dart-v3-tokenizer-241010"


SPECIAL_TOKENS = [
    #
    CommonSpecialTokens.BOS_TOKEN,
    CommonSpecialTokens.EOS_TOKEN,
    CommonSpecialTokens.PAD_TOKEN,
    CommonSpecialTokens.UNK_TOKEN,
    CommonSpecialTokens.SEP_TOKEN,
    CommonSpecialTokens.MASK_TOKEN,
    #
    TagCategoryTokens.RATING_START,
    TagCategoryTokens.RATING_END,
    TagCategoryTokens.GENERAL_START,
    TagCategoryTokens.GENERAL_END,
    TagCategoryTokens.CHARACTER_START,
    TagCategoryTokens.CHARACTER_END,
    TagCategoryTokens.COPYRIGHT_START,
    TagCategoryTokens.COPYRIGHT_END,
    TagCategoryTokens.META_START,
    TagCategoryTokens.META_END,
    #
    RatingTokens.RATING_SFW,
    RatingTokens.RATING_NSFW,
    RatingTokens.RATING_GENERAL,
    RatingTokens.RATING_SENSITIVE,
    RatingTokens.RATING_QUESTIONABLE,
    RatingTokens.RATING_EXPLICIT,
    #
    QualityTokens.QUALITY_BEST,
    QualityTokens.QUALITY_HIGH,
    QualityTokens.QUALITY_NORMAL,
    QualityTokens.QUALITY_LOW,
    QualityTokens.QUALITY_WORST,
    #
    LengthTokens.LENGTH_VERY_SHORT,
    LengthTokens.LENGTH_SHORT,
    LengthTokens.LENGTH_MEDIUM,
    LengthTokens.LENGTH_LONG,
    LengthTokens.LENGTH_VERY_LONG,
    #
    AspectRatioTokens.ASPECT_RATIO_TOO_TALL,
    AspectRatioTokens.ASPECT_RATIO_TALL_WALLPAPER,
    AspectRatioTokens.ASPECT_RATIO_TALL,
    AspectRatioTokens.ASPECT_RATIO_SQUARE,
    AspectRatioTokens.ASPECT_RATIO_WIDE,
    AspectRatioTokens.ASPECT_RATIO_WIDE_WALLPAPER,
    AspectRatioTokens.ASPECT_RATIO_TOO_WIDE,
    #
    InstructionTokens.INPUT_START,
    InstructionTokens.INPUT_END,
    InstructionTokens.GROUP_START,
    InstructionTokens.GROUP_END,
    InstructionTokens.EXAMPLE_START,
    InstructionTokens.EXAMPLE_END,
    InstructionTokens.BAN_START,
    InstructionTokens.BAN_END,
    InstructionTokens.USE_START,
    InstructionTokens.USE_END,
    #
    IdentityTokens.FLAG_KEEP_IDENTITY,
    IdentityTokens.IDENTITY_LEVEL_NONE,
    IdentityTokens.IDENTITY_LEVEL_LAX,
    IdentityTokens.IDENTITY_LEVEL_STRICT,
    #
    MultiModalTokens.IMAGE_START,
    MultiModalTokens.IMAGE_PLACEHOLDER,
    MultiModalTokens.IMAGE_END,
    MultiModalTokens.LINEART_START,
    MultiModalTokens.LINEART_PLACEHOLDER,
    MultiModalTokens.LINEART_END,
    MultiModalTokens.NATURAL_START,
    MultiModalTokens.NATURAL_PLACEHOLDER,
    MultiModalTokens.NATURAL_END,
    MultiModalTokens.TAGGER_START,
    MultiModalTokens.TAGGER_PLACEHOLDER,
    MultiModalTokens.TAGGER_END,
    MultiModalTokens.PROJECTION_START,
    MultiModalTokens.PROJECTION_PLACEHOLDER,
    MultiModalTokens.PROJECTION_END,
    MultiModalTokens.DESCRIBE_START,
    MultiModalTokens.DESCRIBE_PLACEHOLDER,
    MultiModalTokens.DESCRIBE_END,
    #
    TranslationTokens.TRANSLATE_EXACT,
    TranslationTokens.TRANSLATE_APPROX,
    TranslationTokens.TRANSLATE_CREATIVE,
    #
    *RESERVED_TOKENS,
]
assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS))


def prepare_dataset():
    ds = load_dataset(DATASET_REPO_ID, revision=DATASET_REVISION, split=DATASET_SPLIT)
    print(ds)
    return ds


def get_count_map(ds: Dataset, column_name: str):
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


def get_all_tags(ds: Dataset):
    general_tags_count_map = get_count_map(ds, "general")
    copyright_tags_count_map = get_count_map(ds, "copyright")
    character_tags_count_map = get_count_map(ds, "character")
    meta_tags_count_map = get_count_map(ds, "meta")

    popular_general_tags = [
        name for name, count in general_tags_count_map.items() if count > 100
    ]
    popular_copyright_tags = [
        name for name, count in copyright_tags_count_map.items() if count > 100
    ]
    popular_character_tags = [
        name for name, count in character_tags_count_map.items() if count > 100
    ]
    popular_meta_tags = [
        name for name, count in meta_tags_count_map.items() if count > 100
    ]

    return [
        *popular_general_tags,
        *popular_copyright_tags,
        *popular_character_tags,
        *popular_meta_tags,
    ]


def filter_dataset(ds: Dataset):
    def filter_filetype(examples):
        flags = []

        for i, id in enumerate(examples["id"]):
            if examples["file_ext"][i] in [
                "avif",
                "gif",
                "jpg",
                "png",
                "webp",
            ]:
                flags.append(True)
            else:
                flags.append(False)

        return flags

    ds = ds.filter(filter_filetype, batched=True, batch_size=1024, num_proc=16)

    return ds


def train_tokenizer(all_tokens: list[str], special_tokens: list[str]):
    tokenizer = Tokenizer(
        # 頭から順番に番号をつける
        WordLevel(
            vocab={tag: i for i, tag in enumerate(all_tokens)},
            unk_token=CommonSpecialTokens.UNK_TOKEN,
        )
    )
    tokenizer.normalizer = Lowercase()
    tokenizer.pre_tokenizer = Split(
        pattern=Regex(r",(?:\s)*"),
        behavior="removed",
        invert=False,
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
    tokenizer.enable_padding(pad_token=CommonSpecialTokens.PAD_TOKEN)

    return tokenizer


def convert_to_hf_format(tokenizer_object: Tokenizer):
    pretrained_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer_object)

    pretrained_tokenizer.bos_token = CommonSpecialTokens.BOS_TOKEN
    pretrained_tokenizer.eos_token = CommonSpecialTokens.EOS_TOKEN
    pretrained_tokenizer.pad_token = CommonSpecialTokens.PAD_TOKEN
    pretrained_tokenizer.unk_token = CommonSpecialTokens.UNK_TOKEN
    pretrained_tokenizer.sep_token = CommonSpecialTokens.SEP_TOKEN
    pretrained_tokenizer.mask_token = CommonSpecialTokens.MASK_TOKEN

    return pretrained_tokenizer


def main():
    ds = prepare_dataset()
    print(ds)

    ds = filter_dataset(ds)
    print(ds)

    all_tags = get_all_tags(ds)

    all_tokens = [
        *SPECIAL_TOKENS,
        *all_tags,
    ]
    assert len(all_tokens) == len(set(all_tokens)), "Duplicate tokens found!"

    tokenizer = train_tokenizer(all_tokens, SPECIAL_TOKENS)

    print(f"tokenizer: {tokenizer}")
    print(f"tokenizer.get_vocab_size(): {tokenizer.get_vocab_size()}")

    test_text = (
        "<|aspect_ratio:tall|><general>1girl, 2girls, aaa, long hair, very long hair, "
        "<copyright>honkai: star rail, arknights, hatsune miku, "
        "hogeeeeeeeee, <|translate:creative|><|input_end|>, <|pad|>"
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
