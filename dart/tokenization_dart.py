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
    "category_config": "category_config.json",
}

PRETRAINED_VOCAB_FILES_MAP = {
    "category_config": {
        "p1atdev/dart-tokenizer-v1": "https://huggingface.co/p1atdev/dart-tokenizer-v1/resolve/main/tag_category.json"
    }
}


@dataclass
class Category:
    name: str
    bos_token_id: int
    eos_token_id: int


@dataclass
class TagCategoryConfig:
    categories: Dict[str, Category]
    category_to_token_ids: Dict[str, List[int]]


def load_tag_category_config(config_json: str):
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

    def __init__(self, category_config, **kwargs):
        super().__init__(**kwargs)

        self._tokenizer.decoder = Decoder.custom(  # type: ignore
            DartDecoder(list(self.get_added_vocab().keys()))
        )

        self.category_config = load_tag_category_config(category_config)

        self._id_to_category_map = np.zeros(self.vocab_size).astype("uint8")
        for (
            category_id,
            tokens,
        ) in self.category_config.category_to_token_ids.items():
            self._id_to_category_map[tokens] = int(category_id)

    def create_vocab_mask(self, value: int = 1):
        """Create an array of vocab size filled with specified value"""
        return np.full(self.vocab_size, value).astype("uint8")

    def get_token_ids_in_category(self, category_id: Union[int, str]):
        """Get token ids in the specified category"""
        return self.category_config.category_to_token_ids[str(category_id)]

    def get_category(self, category_id: Union[int, str]):
        """Get the specified category config"""
        return self.category_config.categories[str(category_id)]

    def convert_ids_to_category_ids(self, token_ids: Union[int, List[int]]):
        """Get the category ids of specified tokens"""
        return self._id_to_category_map[token_ids]

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
