import logging
import os
import json
from typing import Optional, Dict, List, Tuple, Union
from pydantic.dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from transformers import PreTrainedTokenizerFast
from tokenizers.decoders import Decoder

logger = logging.getLogger(__name__)

VOCAB_FILES_NAMES = {
    "tag_category": "tag_category.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "tag_category": {
        "p1atdev/tokenizer_test_1": "https://huggingface.co/p1atdev/tokenizer_test_1/resolve/main/tag_category.json"
    }
}


@dataclass
class Category:
    name: str
    max_count: Optional[int]
    next_category: List[int]
    can_end: bool
    bos_token_id: int
    eos_token_id: int
    default_mask: int


@dataclass
class SpecialMapping:
    allow: List[int]
    disallow: List[int]


@dataclass
class TagCategoryConfig:
    start_category: int
    categories: Dict[str, Category]
    special_mapping: Dict[
        str, Dict[str, SpecialMapping]
    ]  # {token_id: { category_id: SpecialMapping }}
    category_tags_pairs: Dict[str, List[int]]


class OverrideMask:
    allow: np.ndarray
    disallow: np.ndarray

    def __init__(self, allow: np.ndarray, disallow: np.ndarray) -> None:
        self.allow = allow
        self.disallow = disallow


def load_tag_category(config_json: str):
    with open(config_json, "rb") as file:
        config: TagCategoryConfig = TagCategoryConfig(**json.loads(file.read()))

    return config


class DartDecoder:
    def __init__(self, special_tokens: List[str]):
        self.special_tokens = list(special_tokens)

    def decode_chain(self, tokens: List[str]) -> List[str]:
        new_tokens = []
        is_specials = []

        for i, token in enumerate(tokens):
            is_specials.append(token in self.special_tokens)

            if i == 0:
                new_tokens.append(token)
                continue

            # this token or previous token is special
            if is_specials[i] or is_specials[i - 1]:
                new_tokens.append(token)
                continue

            new_tokens.append(f", {token}")

        return new_tokens


class DartTokenizer(PreTrainedTokenizerFast):
    """Dart tokenizer"""

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP

    def __init__(self, tag_category, **kwargs):
        super().__init__(**kwargs)

        self._tokenizer.decoder = Decoder.custom(  # type: ignore
            DartDecoder(list(self.get_added_vocab().keys()))
        )

        self.tag_category_config = load_tag_category(tag_category)

        self.category_bos_map = {
            category.bos_token_id: category_id
            for category_id, category in self.tag_category_config.categories.items()
        }
        self.category_eos_map = {
            category.eos_token_id: category_id
            for category_id, category in self.tag_category_config.categories.items()
        }

        self._id_to_category_map = np.zeros(self.vocab_size).astype("uint8")
        for category_id, tokens in self.tag_category_config.category_tags_pairs.items():
            self._id_to_category_map[tokens] = int(category_id)

        self.category_mask = self.create_category_vocab_mask()

    def create_vocab_mask(self, value: int = 1):
        """Create an array of vocab size filled with specified value"""
        return np.full(self.vocab_size, value).astype("uint8")

    def create_category_vocab_mask(self):
        """Create vocab masks for each category"""
        return {
            category_id: self.create_vocab_mask(
                value=category.default_mask,
            )
            for category_id, category in self.tag_category_config.categories.items()
        }

    def get_token_ids_in_category(self, category_id: Union[int, str]):
        """Get token ids in the specified category"""
        return self.tag_category_config.category_tags_pairs[str(category_id)]

    def get_category(self, category_id: Union[int, str]):
        """Get the specified category config"""
        return self.tag_category_config.categories[str(category_id)]

    def get_special_mapping(self, token_id: Union[int, str]):
        """Get the special mapping of specified token id"""
        return self.tag_category_config.special_mapping[str(token_id)]

    def get_banned_tokens_mask(self, tokens: Union[str, List[str], int, List[int]]):
        if isinstance(tokens, str):
            tokens = [tokens]
        elif isinstance(tokens, int):
            tokens = [tokens]
        elif isinstance(tokens, list):
            tokens = [  # type: ignore
                self.convert_tokens_to_ids(token) if isinstance(token, str) else token
                for token in tokens
            ]

        assert isinstance(tokens, list) and all(
            [isinstance(token, int) for token in tokens]
        )

        mask = self.create_vocab_mask(value=1)
        mask[tokens] = 0

        return mask

    def convert_ids_to_category_ids(self, token_ids: Union[int, List[int]]):
        return self._id_to_category_map[token_ids]

    def get_next_tokens_mask(
        self,
        input_ids: List[int],
        category_mask: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Get the next token's vocab mask and a category mask"""

        if category_mask == None:
            category_mask = self.category_mask

        vocab_mask = self.create_vocab_mask(value=0)

        if len(input_ids) == 0:
            # only allow bos token
            vocab_mask[self.bos_token_id] = 1

            return vocab_mask, category_mask

        # the last token's id in the input ids
        last_token_id = input_ids[-1]

        if last_token_id == self.unk_token_id:
            # unknown token
            logger.warning(
                "The unk_token was provided! The vocab mask could not be created properly."
            )
            return self.create_vocab_mask(value=1), category_mask

        # if the last token has a special mapping
        if str(last_token_id) in self.tag_category_config.special_mapping.keys():
            for category_id, mapping in self.get_special_mapping(last_token_id).items():
                # update mask
                category_mask[category_id][mapping.allow] = 1
                category_mask[category_id][mapping.disallow] = 0

        if last_token_id == self.bos_token_id:
            # the first category
            start_category_id = self.tag_category_config.start_category
            start_category = self.get_category(start_category_id)

            # only allow the next category's bos token
            vocab_mask[start_category.bos_token_id] = 1

            return vocab_mask, category_mask

        elif last_token_id == self.eos_token_id:
            # end of text. only allows pad token

            vocab_mask[self.pad_token_id] = 1

            return vocab_mask, category_mask

        elif last_token_id in self.category_bos_map:
            # begin of category

            # only allow same category's tags
            current_category_id = self.category_bos_map[last_token_id]
            category = self.get_category(current_category_id)

            tokens_in_category = self.get_token_ids_in_category(current_category_id)
            vocab_mask[tokens_in_category] = 1

            vocab_mask *= category_mask[str(current_category_id)]
            vocab_mask[category.eos_token_id] = 1

            return vocab_mask, category_mask  # current category's mask

        elif last_token_id in self.category_eos_map:
            # boundary of categories

            current_category_id = self.category_eos_map[last_token_id]
            category = self.get_category(current_category_id)

            if category.can_end:
                # this category can finish generation
                vocab_mask[self.eos_token_id] = 1

            for next_category_id in category.next_category:
                # allow the next category's bos token
                vocab_mask[self.get_category(next_category_id).bos_token_id] = 1

            return vocab_mask, category_mask

        else:
            # inside each category
            current_category_id = self.convert_ids_to_category_ids(last_token_id).item()
            tokens_in_category = self.get_token_ids_in_category(current_category_id)

            vocab_mask[tokens_in_category] = 1
            vocab_mask[self.get_category(current_category_id).eos_token_id] = 1
            vocab_mask *= category_mask[str(current_category_id)]
            vocab_mask[input_ids] = 0  # do not reuse used tokens

            return vocab_mask, category_mask
