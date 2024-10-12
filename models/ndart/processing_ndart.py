from dataclasses import dataclass

import torch
import torch.nn as nn

from transformers import (
    AutoModelForTextEncoding,
    AutoModelForCausalLM,
    LlamaForCausalLM,
    BertModel,
    PreTrainedTokenizerBase,
    AutoTokenizer,
    ProcessorMixin,
    GenerationMixin,
    BatchFeature,
    PreTrainedTokenizerFast,
)
from transformers.processing_utils import ProcessingKwargs


class NDartProcessorKwargs:
    _defaults = {
        "natural_kwargs": {
            "padding": True,
        },
        "tag_kwargs": {
            "padding": True,
        },
    }


@dataclass
class NDartProcessingOutput:
    natural: BatchFeature
    tag: BatchFeature


class NDartProcessor(ProcessorMixin):
    attributes = ["natural_tokenizer", "tag_tokenizer"]
    valid_kwargs = ["chat_template", "natural_token"]
    natural_tokenizer_class = "AutoTokenizer"
    tag_tokenizer_class = "AutoTokenizer"

    natural_tokenizer: PreTrainedTokenizerBase
    tag_tokenizer: PreTrainedTokenizerBase

    def __init__(
        self,
        natural_tokenizer=None,
        tag_tokenizer=None,
        natural_token="<|natural|>",
        chat_template=None,
        **kwargs,
    ):
        self.natural_token = natural_token
        super().__init__(
            natural_tokenizer, tag_tokenizer, chat_template=chat_template, **kwargs
        )

    def __call__(
        self,
        natural_text: str | list[str] | None = None,
        tag_text: str | list[str] | None = None,
        **kwargs,
    ) -> NDartProcessingOutput:
        if tag_text is None:
            raise ValueError("tag_text is required for NDartProcessor")

        def _validate_text_input(text) -> str | list[str]:
            if isinstance(text, list):
                assert all(
                    isinstance(t, str) for t in text
                ), f"Expected list of str but got {type(text)}"
                assert all(len(t) > 0 for t in text), "Expected non-empty strings"
            else:
                assert isinstance(text, str), f"Expected str but got {type(text)}"
                assert len(text) > 0, "Expected non-empty string"
            return text

        def _normalize_text_input(text: str | list[str]) -> list[str]:
            if isinstance(text, str):
                return [text]
            return text

        natural_text: str | list[str] = _validate_text_input(natural_text)
        natural_text = _normalize_text_input(natural_text)
        tag_text: str | list[str] = _validate_text_input(tag_text)
        tag_text = _normalize_text_input(tag_text)

        natural_output_kwargs = {
            **NDartProcessorKwargs._defaults["natural_kwargs"],
            **kwargs,
        }
        tag_output_kwargs = {
            **NDartProcessorKwargs._defaults["tag_kwargs"],
            **kwargs,
        }

        natural_tokens = self.natural_tokenizer(
            natural_text,
            **natural_output_kwargs,
        )
        tag_tokens = self.tag_tokenizer(
            tag_text,
            **tag_output_kwargs,
        )

        return NDartProcessingOutput(
            natural=BatchFeature(data={**natural_tokens}),
            tag=BatchFeature(data={**tag_tokens}),
        )

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->PreTrained
    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tag_tokenizer.batch_decode(*args, **kwargs)

    # Copied from transformers.models.clip.processing_clip.CLIPProcessor.decode with CLIP->PreTrained
    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizerFast's [`~PreTrainedTokenizer.decode`]. Please refer to
        the docstring of this method for more information.
        """
        return self.tag_tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        return ["natural_text", "tag_text"]


if __name__ == "__main__":
    natural_tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")
    tag_tokenizer = AutoTokenizer.from_pretrained("p1atdev/dart-v3-tokenizer-241010")
    processor = NDartProcessor(
        natural_tokenizer=natural_tokenizer,
        tag_tokenizer=tag_tokenizer,
    )
    output = processor("Hello, world!", "<|natural|><general>1girl, solo")
    print(output)
